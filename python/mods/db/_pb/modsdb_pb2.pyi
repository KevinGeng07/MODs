from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ColumnType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COLUMN_TYPE_UNSPECIFIED: _ClassVar[ColumnType]
    INT64: _ClassVar[ColumnType]
    FLOAT64: _ClassVar[ColumnType]
    TENSOR: _ClassVar[ColumnType]

class DType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    F32: _ClassVar[DType]
    U8: _ClassVar[DType]
COLUMN_TYPE_UNSPECIFIED: ColumnType
INT64: ColumnType
FLOAT64: ColumnType
TENSOR: ColumnType
F32: DType
U8: DType

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Tensor(_message.Message):
    __slots__ = ("shape", "data", "dtype")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    dtype: DType
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ..., dtype: _Optional[_Union[DType, str]] = ...) -> None: ...

class Value(_message.Message):
    __slots__ = ("i", "f", "t")
    I_FIELD_NUMBER: _ClassVar[int]
    F_FIELD_NUMBER: _ClassVar[int]
    T_FIELD_NUMBER: _ClassVar[int]
    i: int
    f: float
    t: Tensor
    def __init__(self, i: _Optional[int] = ..., f: _Optional[float] = ..., t: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class Column(_message.Message):
    __slots__ = ("name", "type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: ColumnType
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[ColumnType, str]] = ...) -> None: ...

class Schema(_message.Message):
    __slots__ = ("columns",)
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    columns: _containers.RepeatedCompositeFieldContainer[Column]
    def __init__(self, columns: _Optional[_Iterable[_Union[Column, _Mapping]]] = ...) -> None: ...

class Row(_message.Message):
    __slots__ = ("id", "version", "cols")
    class ColsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    COLS_FIELD_NUMBER: _ClassVar[int]
    id: int
    version: int
    cols: _containers.MessageMap[str, Value]
    def __init__(self, id: _Optional[int] = ..., version: _Optional[int] = ..., cols: _Optional[_Mapping[str, Value]] = ...) -> None: ...

class Rows(_message.Message):
    __slots__ = ("rows",)
    ROWS_FIELD_NUMBER: _ClassVar[int]
    rows: _containers.RepeatedCompositeFieldContainer[Row]
    def __init__(self, rows: _Optional[_Iterable[_Union[Row, _Mapping]]] = ...) -> None: ...

class CreateTableRequest(_message.Message):
    __slots__ = ("table", "schema")
    TABLE_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    table: str
    schema: Schema
    def __init__(self, table: _Optional[str] = ..., schema: _Optional[_Union[Schema, _Mapping]] = ...) -> None: ...

class DropTableRequest(_message.Message):
    __slots__ = ("table",)
    TABLE_FIELD_NUMBER: _ClassVar[int]
    table: str
    def __init__(self, table: _Optional[str] = ...) -> None: ...

class GetRowsRequest(_message.Message):
    __slots__ = ("table", "ids", "columns")
    TABLE_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    table: str
    ids: _containers.RepeatedScalarFieldContainer[int]
    columns: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, table: _Optional[str] = ..., ids: _Optional[_Iterable[int]] = ..., columns: _Optional[_Iterable[str]] = ...) -> None: ...

class ScanRequest(_message.Message):
    __slots__ = ("table", "columns", "filter", "status", "epoch")
    TABLE_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    table: str
    columns: _containers.RepeatedScalarFieldContainer[str]
    filter: bool
    status: int
    epoch: int
    def __init__(self, table: _Optional[str] = ..., columns: _Optional[_Iterable[str]] = ..., filter: _Optional[bool] = ..., status: _Optional[int] = ..., epoch: _Optional[int] = ...) -> None: ...

class Op(_message.Message):
    __slots__ = ("table", "id", "kind", "cols")
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        KIND_UNSPECIFIED: _ClassVar[Op.Kind]
        PUT: _ClassVar[Op.Kind]
        PATCH: _ClassVar[Op.Kind]
        INCR: _ClassVar[Op.Kind]
    KIND_UNSPECIFIED: Op.Kind
    PUT: Op.Kind
    PATCH: Op.Kind
    INCR: Op.Kind
    class ColsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    TABLE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    COLS_FIELD_NUMBER: _ClassVar[int]
    table: str
    id: int
    kind: Op.Kind
    cols: _containers.MessageMap[str, Value]
    def __init__(self, table: _Optional[str] = ..., id: _Optional[int] = ..., kind: _Optional[_Union[Op.Kind, str]] = ..., cols: _Optional[_Mapping[str, Value]] = ...) -> None: ...

class WriteBatch(_message.Message):
    __slots__ = ("ops", "tag")
    OPS_FIELD_NUMBER: _ClassVar[int]
    TAG_FIELD_NUMBER: _ClassVar[int]
    ops: _containers.RepeatedCompositeFieldContainer[Op]
    tag: str
    def __init__(self, ops: _Optional[_Iterable[_Union[Op, _Mapping]]] = ..., tag: _Optional[str] = ...) -> None: ...

class CommitAck(_message.Message):
    __slots__ = ("lsn",)
    LSN_FIELD_NUMBER: _ClassVar[int]
    lsn: int
    def __init__(self, lsn: _Optional[int] = ...) -> None: ...

class FetchAndMarkRequest(_message.Message):
    __slots__ = ("table", "ids", "mark", "columns", "tag")
    class MarkEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    TABLE_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    MARK_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    TAG_FIELD_NUMBER: _ClassVar[int]
    table: str
    ids: _containers.RepeatedScalarFieldContainer[int]
    mark: _containers.MessageMap[str, Value]
    columns: _containers.RepeatedScalarFieldContainer[str]
    tag: str
    def __init__(self, table: _Optional[str] = ..., ids: _Optional[_Iterable[int]] = ..., mark: _Optional[_Mapping[str, Value]] = ..., columns: _Optional[_Iterable[str]] = ..., tag: _Optional[str] = ...) -> None: ...

class CreateCheckpointRequest(_message.Message):
    __slots__ = ("name", "blob")
    NAME_FIELD_NUMBER: _ClassVar[int]
    BLOB_FIELD_NUMBER: _ClassVar[int]
    name: str
    blob: bytes
    def __init__(self, name: _Optional[str] = ..., blob: _Optional[bytes] = ...) -> None: ...

class Checkpoint(_message.Message):
    __slots__ = ("id", "name", "lsn", "created_unix_ms")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    LSN_FIELD_NUMBER: _ClassVar[int]
    CREATED_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    lsn: int
    created_unix_ms: int
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., lsn: _Optional[int] = ..., created_unix_ms: _Optional[int] = ...) -> None: ...

class CheckpointList(_message.Message):
    __slots__ = ("checkpoints",)
    CHECKPOINTS_FIELD_NUMBER: _ClassVar[int]
    checkpoints: _containers.RepeatedCompositeFieldContainer[Checkpoint]
    def __init__(self, checkpoints: _Optional[_Iterable[_Union[Checkpoint, _Mapping]]] = ...) -> None: ...

class RestoreCheckpointRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class RestoreCheckpointResponse(_message.Message):
    __slots__ = ("checkpoint", "blob", "lsn")
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    BLOB_FIELD_NUMBER: _ClassVar[int]
    LSN_FIELD_NUMBER: _ClassVar[int]
    checkpoint: Checkpoint
    blob: bytes
    lsn: int
    def __init__(self, checkpoint: _Optional[_Union[Checkpoint, _Mapping]] = ..., blob: _Optional[bytes] = ..., lsn: _Optional[int] = ...) -> None: ...

class StatsResponse(_message.Message):
    __slots__ = ("last_lsn", "last_snapshot_lsn", "wal_bytes", "wal_segments", "commits", "fsyncs")
    LAST_LSN_FIELD_NUMBER: _ClassVar[int]
    LAST_SNAPSHOT_LSN_FIELD_NUMBER: _ClassVar[int]
    WAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    WAL_SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    COMMITS_FIELD_NUMBER: _ClassVar[int]
    FSYNCS_FIELD_NUMBER: _ClassVar[int]
    last_lsn: int
    last_snapshot_lsn: int
    wal_bytes: int
    wal_segments: int
    commits: int
    fsyncs: int
    def __init__(self, last_lsn: _Optional[int] = ..., last_snapshot_lsn: _Optional[int] = ..., wal_bytes: _Optional[int] = ..., wal_segments: _Optional[int] = ..., commits: _Optional[int] = ..., fsyncs: _Optional[int] = ...) -> None: ...
