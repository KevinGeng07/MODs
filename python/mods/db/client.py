"""Python client for modsdb.

The gRPC channel is opened lazily per process: DataLoader workers are separate
processes and each needs its own connection. Pickling a Client (as happens
when a Dataset is sent to a spawned worker) carries only the address."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

# Workers are spawned (fork+exec); silence gRPC's fork handlers. Must precede `import grpc`.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "false")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import grpc  # noqa: E402

from ._pb import modsdb_pb2 as pb  # noqa: E402
from ._pb import modsdb_pb2_grpc as rpc  # noqa: E402

IDLE, QUEUED, CONSUMED = 0, 1, 2
_TYPES = {"int64": pb.INT64, "float64": pb.FLOAT64}


class DBError(Exception):
    pass


def to_value(v: Any) -> pb.Value:
    if isinstance(v, (bool, int)):
        return pb.Value(i=int(v))
    if isinstance(v, float):
        return pb.Value(f=v)
    raise TypeError(f"cannot store {type(v).__name__}")


def from_value(v: pb.Value) -> Any:
    kind = v.WhichOneof("v")
    return getattr(v, kind) if kind else None


class Row(dict):
    """A row's columns as a dict, plus .id and .version (LSN of last write)."""

    def __init__(self, r: pb.Row):
        super().__init__({k: from_value(v) for k, v in r.cols.items()})
        self.id, self.version = r.id, r.version


def _cols(d: dict) -> dict:
    return {k: to_value(v) for k, v in d.items()}


def put(table: str, id: int, cols: dict) -> pb.Op:
    return pb.Op(table=table, id=id, kind=pb.Op.PUT, cols=_cols(cols))


def patch(table: str, id: int, cols: dict) -> pb.Op:
    return pb.Op(table=table, id=id, kind=pb.Op.PATCH, cols=_cols(cols))


def incr(table: str, id: int, cols: dict) -> pb.Op:
    return pb.Op(table=table, id=id, kind=pb.Op.INCR, cols=_cols(cols))


class Client:
    def __init__(self, addr: str = "127.0.0.1:7070"):
        self.addr = addr
        self._pid: Optional[int] = None
        self._stub = None

    def __getstate__(self):
        return {"addr": self.addr}

    def __setstate__(self, s):
        self.__init__(s["addr"])

    def _call(self, method: str, req):
        if self._pid != os.getpid():
            # Starting a run writes one state row per training sample in a single
            # commit (about 10 MB for MNIST), above gRPC's 4 MB default.
            opts = [("grpc.max_send_message_length", 256 << 20), ("grpc.max_receive_message_length", 256 << 20)]
            self._stub = rpc.ModsDBStub(grpc.insecure_channel(self.addr, options=opts))
            self._pid = os.getpid()
        try:
            return getattr(self._stub, method)(req, timeout=60)
        except grpc.RpcError as e:
            raise DBError(f"{e.code().name}: {e.details()}") from None

    def create_table(self, name: str, schema: dict[str, str]) -> int:
        cols = [pb.Column(name=n, type=_TYPES[t]) for n, t in schema.items()]
        return self._call("CreateTable", pb.CreateTableRequest(table=name, schema=pb.Schema(columns=cols))).lsn

    def drop_table(self, name: str) -> int:
        return self._call("DropTable", pb.DropTableRequest(table=name)).lsn

    def get_rows(self, table: str, ids: Sequence[int], columns: Sequence[str] = ()) -> list[Row]:
        return [Row(r) for r in self._call("GetRows", pb.GetRowsRequest(table=table, ids=ids, columns=columns)).rows]

    def scan(self, table: str, columns: Sequence[str] = (), status: Optional[int] = None, epoch: int = 0) -> list[Row]:
        req = pb.ScanRequest(table=table, columns=columns, filter=status is not None, status=status or 0, epoch=epoch)
        return [Row(r) for r in self._call("Scan", req).rows]

    def apply(self, ops: Sequence[pb.Op], tag: str = "") -> int:
        return self._call("Apply", pb.WriteBatch(ops=ops, tag=tag)).lsn

    def fetch_and_mark(self, table: str, ids: Sequence[int], mark: dict, columns: Sequence[str] = (), tag: str = "") -> list[Row]:
        req = pb.FetchAndMarkRequest(table=table, ids=ids, mark=_cols(mark), columns=columns, tag=tag)
        return [Row(r) for r in self._call("FetchAndMark", req).rows]

    def stats(self) -> dict:
        s = self._call("Stats", pb.Empty())
        return {f.name: getattr(s, f.name) for f in s.DESCRIPTOR.fields}
