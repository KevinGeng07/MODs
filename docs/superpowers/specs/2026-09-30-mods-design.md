# MODs — Model Data Observability: Design Spec

**Date:** 2026-09-30
**Status:** Draft, awaiting review

> **Update 2026-09-30: cut down to an MVP.** The code now implements a subset of this spec.
> Removed: the change-stream hub and `Subscribe` (the UI polls instead), the MANIFEST (the
> catalogue is the directory listing), the status index, fsync modes, BulkLoad/Delete/
> expect_version/pagination, stale-edit detection, table editing, the loss chart, the change
> log and `ReadAt`. The current API is `db/proto/modsdb/v1/modsdb.proto`.


## 1. Goal

A local platform for watching a small PyTorch model train, where the **training data lives in a
custom database (written in Go) that records every step of the feeding process**. The database is
the main subject of the project: write-ahead log (WAL), snapshots, checkpoints, crash recovery,
concurrent writers, and a live change stream.

The user can:

1. See a static graph of the model's computation (torch.fx dataflow graph, with tensor shapes).
2. See which rows each DataLoader worker has prefetched and queued, updated live from the database.
3. Train in **Run** mode (continuous, speed adjustable) or **Step** mode (one batch per click).
4. Choose a validation sample, **Evaluate** it at any time, and **Shuffle** to a different one.
5. **Save** and **Restore** a full session checkpoint, then resume training on exactly the same next
   batch.

### Constraints

- CPU only. The model and dataset are small: ≤ ~100k rows, a few KB per row. The whole dataset
  fits in RAM.
- Everything runs on one machine: the Go DB service, the Python backend, and the browser UI.
- The DataLoader can use multiple worker processes, and each one talks to the DB on its own.
- Dependencies found on this machine: Go 1.27, Python with torch 2.9.1, grpcio, fastapi, and
  scikit-learn. `protoc`/`buf` must be installed to generate code (`brew install bufbuild/buf/buf`).

### Non-goals (YAGNI)

- No SQL, query planner, or secondary indexes (beyond one status index, §4.6).
- No interactive multi-statement transactions. A single `WriteBatch` is the unit of atomicity.
- No replication, authentication, or network exposure beyond `localhost`.
- No GPU support, distributed training, or datasets larger than RAM.

## 2. Architecture

```
┌──────────────┐  HTTP + WebSocket   ┌────────────────────────────────┐
│  Browser UI  │ ◀─────────────────▶ │  Python backend (FastAPI)      │
│ graph/queues │                     │  TrainingSession (one thread)  │
│ table/ctrls  │                     │  ├─ model, optimizer, sampler  │
└──────────────┘                     │  └─ DataLoader ─┬─ worker 0 ─┐ │
                                     └──────────┬──────┴─ worker N ─┤─┘
                                                │ gRPC (localhost)   │ gRPC (one channel per worker)
                                                ▼                    ▼
                                     ┌────────────────────────────────┐
                                     │  modsdb (Go)                   │
                                     │  gRPC server → Engine          │
                                     │   ├─ commit loop (one writer)  │
                                     │   ├─ memtable (immutable rows) │
                                     │   ├─ WAL segments              │
                                     │   ├─ snapshots + MANIFEST      │
                                     │   └─ change-stream hub         │
                                     └────────────────────────────────┘
```

- The browser **never talks to Go directly**. The Python backend relays DB change events to the
  UI over a WebSocket.
- The DB is the **single source of truth** for sample state. The queue view in the UI is a live
  view of rows with `status = QUEUED`. It is not a separate bookkeeping structure in Python.

### Repository layout

```
db/                         Go module: github.com/<you>/mods/db
  cmd/modsdb/main.go        flags: --data-dir, --addr, --fsync, --snapshot-every
  proto/modsdb/v1/modsdb.proto
  internal/wal/             segment writer/reader, record framing, CRC
  internal/snapshot/        snapshot writer/reader
  internal/manifest/        checkpoint + snapshot catalogue
  internal/engine/          memtable, commit loop, recovery, apply ops
  internal/hub/             change-stream fan-out
  internal/server/          gRPC handlers (thin, no logic)
python/mods/
  db/client.py              typed wrapper over generated stubs
  db/fake.py                in-memory fake store for Python unit tests
  data.py                   DBDataset, epoch-aware sampler
  graph.py                  fx trace → JSON
  session.py                TrainingSession (run/step/eval/checkpoint)
  server/app.py             FastAPI routes + WebSocket
  server/static/            index.html, app.js, styles.css
examples/digits_mlp.py      seeds DB with sklearn digits, launches everything
```

## 3. Data model

### 3.1 Types

Each table has a fixed schema, declared when the table is created. Column types:

| Type      | Go representation              | Notes                                 |
|-----------|--------------------------------|---------------------------------------|
| `INT64`   | `int64`                        |                                       |
| `FLOAT64` | `float64`                      |                                       |
| `STRING`  | `string`                       |                                       |
| `BYTES`   | `[]byte`                       |                                       |
| `TENSOR`  | `{dtype, shape []int64, data}` | raw little-endian; Python converts it with `torch.frombuffer` |

Every row has a primary key `id uint64` and a system column `version` (the LSN of the row's last
write), which the engine maintains.

### 3.2 Tables used by the platform

**`train` and `val`** (same schema):

| Column         | Type    | Written by                        |
|----------------|---------|-----------------------------------|
| `x`            | TENSOR  | seed script; user edits           |
| `y`            | INT64   | seed script; user relabels        |
| `epoch`        | INT64   | worker on fetch (epoch of the current status) |
| `status`       | INT64   | enum `IDLE=0, QUEUED=1, CONSUMED=2` |
| `worker_id`    | INT64   | worker on fetch                   |
| `batch_idx`    | INT64   | worker on fetch (global batch number in the epoch) |
| `times_seen`   | INT64   | main process on consume (increment) |
| `last_loss`    | FLOAT64 | main process on consume           |
| `last_pred`    | INT64   | main process on consume           |
| `last_step`    | INT64   | main process on consume           |
| `fetched_version` | INT64 | worker on fetch: the row version the worker actually read |

**Epoch trick (avoids rewriting every row each epoch):** a row's `status` only counts if
`row.epoch == session.epoch`. Otherwise the row is treated as `IDLE`. Starting a new epoch is
therefore one write to `session`, not N row resets.

**`session`**: a single row (`id = 1`) with columns `epoch`, `global_step`, `val_sample_id`, and
`sampler_seed` (all `INT64`). It is part of every snapshot.

**`eval_log`**: an append-only table with `step, epoch, val_id, loss, pred, target, probs(TENSOR)`.

## 4. Storage engine (Go)

### 4.1 Memtable

- `map[tableName]*Table`, and each `Table` has `rows map[uint64]*Row`.
- **Rows are immutable.** An update allocates a new `*Row` and swaps the pointer. So a snapshot
  can clone the maps (shallow, pointers only) under a short read lock, then serialize in the
  background while writes continue.
- Reads use a `sync.RWMutex` per table. Writes happen only in the commit loop (§4.3).

### 4.2 WAL

- Directory `wal/`, segment files `000000000001.log`, rotated at 64 MiB.
- Record framing (little-endian):

  ```
  ┌─────────┬─────────┬─────────┬──────┬───────────────┐
  │ len u32 │ crc u32 │ lsn u64 │ type │ payload (len) │
  └─────────┴─────────┴─────────┴──────┴───────────────┘
  crc = CRC32C(lsn ‖ type ‖ payload)
  ```

- Record types: `CREATE_TABLE`, `DROP_TABLE`, `WRITE_BATCH`, `CHECKPOINT`, `RESTORE`.
  The payload is the protobuf encoding of the matching message from the gRPC API, so there is
  one encoding shared by the wire and the disk.
- LSNs are strictly increasing and never reused, including across restores.
- **fsync policy** (flag `--fsync`):
  - `commit` (default): **group commit**. The commit loop drains every pending request, writes
    all of their records, then runs one `fsync` before replying to all of them.
  - `interval=10ms`: reply immediately; a background ticker runs `fsync`. A crash can lose up to
    one interval of writes.
  - `off`: for benchmarks only.

### 4.3 Commit loop (single writer)

```
for req := range commitCh (drain up to 512 or until empty):
    validate each op against schema + current state   → reject that request, not the group
    assign LSN; encode record; append to WAL buffer
flush + fsync (per policy)
apply all accepted ops to memtable (new row pointers, version = LSN)
publish ChangeEvents to hub
reply to each request with its LSN (or error)
```

- The client sees success only after the record is durable, according to the fsync policy.
- A request is **all-or-nothing**. If one op fails validation, nothing in that request is applied.
- **Read-your-writes:** changes are applied to the memtable before the reply, so any read issued
  after the reply sees them.

### 4.4 Snapshots

- File `snap/<lsn>.snap`, written as `tmp` → `fsync` → `rename` → `fsync(dir)`.
- Format: header `{magic "MODSNAP1", format_version, lsn, n_tables}`, then for each table its
  schema followed by length-delimited protobuf `Row`s, then a footer with CRC32C of the body.
- **Automatic snapshots** run every `--snapshot-every` bytes of WAL (default 32 MiB) and on clean
  shutdown. The two most recent automatic snapshots are kept.
- **WAL truncation:** a segment is deleted once every record in it has an LSN ≤ the newest
  automatic snapshot's LSN, and (if time travel is enabled, §4.8) it falls outside the retention
  window.

### 4.5 Checkpoints (save state) and restore

A **checkpoint** has three parts: a snapshot at LSN L, a manifest entry, and an opaque
`user_blob` supplied by Python. The blob holds model weights, optimizer state, and sampler state,
produced with `torch.save` into bytes. All three are committed together.

- `MANIFEST` is an append-only log of `{op: ADD_CHECKPOINT|DEL_CHECKPOINT|ADD_SNAPSHOT|DEL_SNAPSHOT, ...}`.
  At startup it is replayed to build the catalogue, and it is compacted when it grows past 1 MiB.
- Checkpoint snapshots are **pinned**: automatic retention never deletes them. They are removed
  only by `DeleteCheckpoint`.
- Order for `CreateCheckpoint`: go through the commit loop to get LSN L and write a `CHECKPOINT`
  WAL record, then write the snapshot at L, then write `blob/<id>.bin`, then append to the
  manifest. The checkpoint exists only when the manifest entry is written. After a crash, files
  that no manifest entry refers to are garbage-collected.
- `RestoreCheckpoint(id)`:
  1. Stop the commit loop from accepting new requests (requests already queued finish first).
  2. Load the checkpoint snapshot into a fresh memtable.
  3. Write a `RESTORE{checkpoint_id, snapshot_lsn}` record at a new LSN R > every earlier LSN.
  4. Immediately take an automatic snapshot at R. Recovery then never has to replay across a
     restore; the marker is kept only for auditing.
  5. Swap in the new memtable, and send a `RESET` event to every subscriber.
  6. Return `user_blob` and R.

### 4.6 Status index

Each table keeps `map[status]map[id]struct{}`, updated in the apply step. This lets the queue
panel run `Scan(status=QUEUED)` without a full scan. It is the only index.

### 4.7 Recovery (startup)

1. Replay `MANIFEST`, then load the newest valid automatic snapshot (checking its CRC). If the
   newest is corrupt, fall back to the next one.
2. Replay WAL records with LSN > snapshot LSN, in order.
3. Stop at the first short or CRC-invalid record. If it is in the **last** segment, truncate the
   torn tail and continue. If it is in an earlier segment, **refuse to start** and report
   corruption (with a `--force-recover` flag that truncates there).
4. Garbage-collect files not referenced by the manifest, then open for traffic.

### 4.8 Time travel (phase 5, optional)

`ReadAt(table, ids, lsn)` finds the snapshot with the largest LSN ≤ `lsn`, replays WAL up to
`lsn` into a scratch memtable, and answers from it. This needs `--wal-retention` (for example
2 GiB) so that old segments are kept. It powers a UI scrubber that shows the table as of step K.

### 4.9 Change-stream hub

- The hub keeps a ring buffer of the last 65,536 committed `ChangeEvent`s.
- `Subscribe(from_lsn)`: if `from_lsn` is still in the ring, the hub sends the backlog and then
  live events. Otherwise it sends `RESET` and the client re-scans.
- Each subscriber has a bounded channel (4,096 events). A subscriber that falls behind gets
  `LAGGED` and is disconnected, so the writer never blocks. It resubscribes from its last LSN.

## 5. gRPC API (`modsdb.v1`)

```proto
syntax = "proto3";
package modsdb.v1;

service ModsDB {
  // ---- schema ----
  rpc CreateTable(CreateTableRequest)   returns (CommitAck);
  rpc DropTable(DropTableRequest)       returns (CommitAck);
  rpc ListTables(ListTablesRequest)     returns (ListTablesResponse);

  // ---- reads ----
  rpc GetRows(GetRowsRequest)           returns (GetRowsResponse);    // batch point-lookup
  rpc Scan(ScanRequest)                 returns (ScanResponse);       // paged, optional status filter
  rpc Count(CountRequest)               returns (CountResponse);

  // ---- writes ----
  rpc Apply(WriteBatch)                 returns (CommitAck);          // atomic multi-op
  rpc BulkLoad(stream BulkLoadChunk)    returns (CommitAck);          // seeding; one WAL record per chunk
  rpc FetchAndMark(FetchAndMarkRequest) returns (GetRowsResponse);    // atomic read + patch (worker fetch)

  // ---- change stream ----
  rpc Subscribe(SubscribeRequest)       returns (stream ChangeEvent);

  // ---- durability / save state ----
  rpc CreateCheckpoint(CreateCheckpointRequest)   returns (Checkpoint);
  rpc ListCheckpoints(ListCheckpointsRequest)     returns (ListCheckpointsResponse);
  rpc RestoreCheckpoint(RestoreCheckpointRequest) returns (RestoreCheckpointResponse);
  rpc DeleteCheckpoint(DeleteCheckpointRequest)   returns (CommitAck);
  rpc ForceSnapshot(ForceSnapshotRequest)         returns (SnapshotInfo);

  // ---- introspection (for the UI's DB panel) ----
  rpc Stats(StatsRequest)               returns (StatsResponse);
  rpc ReadAt(ReadAtRequest)             returns (GetRowsResponse);    // phase 5
}

// ---------- values ----------
enum ColumnType { COLUMN_TYPE_UNSPECIFIED = 0; INT64 = 1; FLOAT64 = 2; STRING = 3; BYTES = 4; TENSOR = 5; }
enum DType      { DTYPE_UNSPECIFIED = 0; F32 = 1; F64 = 2; I64 = 3; U8 = 4; }

message Tensor { DType dtype = 1; repeated int64 shape = 2; bytes data = 3; }
message Value {
  oneof v { int64 i = 1; double f = 2; string s = 3; bytes b = 4; Tensor t = 5; }
}
message Column { string name = 1; ColumnType type = 2; bool nullable = 3; }
message Schema { repeated Column columns = 1; }
message Row {
  uint64 id = 1;
  uint64 version = 2;                 // LSN of last write, set by server
  map<string, Value> cols = 3;
}

// ---------- schema ----------
message CreateTableRequest { string table = 1; Schema schema = 2; }
message DropTableRequest   { string table = 1; }
message ListTablesRequest  {}
message TableInfo          { string name = 1; Schema schema = 2; uint64 row_count = 3; }
message ListTablesResponse { repeated TableInfo tables = 1; }

// ---------- reads ----------
message GetRowsRequest  { string table = 1; repeated uint64 ids = 2; repeated string columns = 3; } // empty columns = all
message GetRowsResponse { repeated Row rows = 1; uint64 read_lsn = 2; repeated uint64 missing_ids = 3; }

message StatusFilter { int64 status = 1; int64 epoch = 2; }   // matches status AND epoch
message ScanRequest {
  string table = 1;
  uint64 after_id = 2;               // keyset pagination
  uint32 limit = 3;                  // server caps at 1000
  repeated string columns = 4;
  StatusFilter filter = 5;           // optional; uses the status index
}
message ScanResponse  { repeated Row rows = 1; uint64 next_after_id = 2; bool done = 3; uint64 read_lsn = 4; }
message CountRequest  { string table = 1; StatusFilter filter = 2; }
message CountResponse { uint64 count = 1; }

// ---------- writes ----------
message Put       { uint64 id = 1; map<string, Value> cols = 2; }           // insert or full replace
message Patch     { uint64 id = 1; map<string, Value> set = 2; }            // set some columns
message Increment { uint64 id = 1; string column = 2; Value delta = 3; }    // INT64/FLOAT64 only
message Delete    { uint64 id = 1; }
message Op {
  string table = 1;
  uint64 expect_version = 2;        // 0 = unconditional; else fail if row.version != this
  oneof kind { Put put = 3; Patch patch = 4; Increment incr = 5; Delete del = 6; }
}
message WriteBatch { repeated Op ops = 1; string client_tag = 2; }          // tag shows in change stream
message CommitAck  { uint64 lsn = 1; }

message BulkLoadChunk { string table = 1; repeated Put rows = 2; }

message FetchAndMarkRequest {
  string table = 1;
  repeated uint64 ids = 2;
  map<string, Value> mark = 3;      // e.g. {status:QUEUED, epoch:3, worker_id:1, batch_idx:17}
  repeated string columns = 4;      // columns to return (read BEFORE the mark is applied)
  bool record_fetched_version = 5;  // also set fetched_version = pre-mark version
  string client_tag = 6;
}

// ---------- change stream ----------
message SubscribeRequest { uint64 from_lsn = 1; repeated string tables = 2; repeated string columns = 3; }
message RowChange {
  string table = 1; uint64 id = 2;
  enum Kind { KIND_UNSPECIFIED = 0; UPSERT = 1; DELETE = 2; }
  Kind kind = 3;
  map<string, Value> changed = 4;   // only columns that changed (filtered by SubscribeRequest.columns)
}
message ChangeEvent {
  uint64 lsn = 1;
  string client_tag = 2;
  oneof body {
    WriteBatchEvent batch = 3;
    Control control = 4;
  }
}
message WriteBatchEvent { repeated RowChange changes = 1; }
message Control {
  enum Kind { KIND_UNSPECIFIED = 0; RESET = 1; LAGGED = 2; SCHEMA_CHANGED = 3; CHECKPOINT_CREATED = 4; RESTORED = 5; }
  Kind kind = 1;
  string detail = 2;
}

// ---------- durability ----------
message CreateCheckpointRequest   { string name = 1; bytes user_blob = 2; }
message Checkpoint                { string id = 1; string name = 2; uint64 lsn = 3; int64 created_unix_ms = 4; uint64 blob_bytes = 5; uint64 snapshot_bytes = 6; }
message ListCheckpointsRequest    {}
message ListCheckpointsResponse   { repeated Checkpoint checkpoints = 1; }
message RestoreCheckpointRequest  { string id = 1; }
message RestoreCheckpointResponse { Checkpoint checkpoint = 1; bytes user_blob = 2; uint64 restore_lsn = 3; }
message DeleteCheckpointRequest   { string id = 1; }
message ForceSnapshotRequest      {}
message SnapshotInfo              { uint64 lsn = 1; uint64 bytes = 2; int64 duration_us = 3; }

// ---------- introspection ----------
message StatsRequest {}
message StatsResponse {
  uint64 last_lsn = 1;
  uint64 durable_lsn = 2;
  uint64 wal_bytes = 3;
  uint32 wal_segments = 4;
  uint64 last_snapshot_lsn = 5;
  double commits_per_sec = 6;
  double avg_group_size = 7;          // requests per fsync
  double p50_commit_us = 8;
  double p99_commit_us = 9;
  uint32 subscribers = 10;
}
message ReadAtRequest { string table = 1; repeated uint64 ids = 2; uint64 lsn = 3; }
```

### Error mapping (gRPC status codes)

| Condition                                  | Code                  |
|--------------------------------------------|-----------------------|
| Unknown table/checkpoint                   | `NOT_FOUND`           |
| Schema/type mismatch, bad increment column | `INVALID_ARGUMENT`    |
| `expect_version` mismatch                  | `ABORTED` (client may retry) |
| Put on existing id via `BulkLoad`          | `ALREADY_EXISTS`      |
| Restore in progress                        | `UNAVAILABLE` (client retries with backoff) |
| WAL write/fsync failure                    | `INTERNAL`, and the server goes **read-only** until restarted |
| `ReadAt` LSN outside the retention window  | `OUT_OF_RANGE`        |

## 6. Python side

### 6.1 `mods.db.Client`

A thin typed wrapper over the gRPC stubs, with `torch.Tensor` ⇄ `Tensor` conversion. The channel
is created **lazily, per process**, because gRPC channels are not fork-safe. DataLoaders use
`multiprocessing_context="spawn"`, which is already the default on macOS.
`mods.db.FakeStore` implements the same methods in memory, for fast unit tests.

### 6.2 `DBDataset` and sampler

- `DBDataset(table, client_factory)` implements `__getitems__(ids)` (supported in torch 2.x),
  so each worker makes **one `FetchAndMark` RPC per batch**, not one per sample. The call marks
  rows `{status: QUEUED, epoch, worker_id, batch_idx}` and returns `x, y, version`.
- `worker_id` comes from `torch.utils.data.get_worker_info()`.
- `EpochSampler(n, seed)` is a deterministic permutation built from
  `torch.Generator().manual_seed(seed + epoch)`. It exposes `state_dict()`/`load_state_dict()`
  holding `{seed, epoch, next_batch}`, and a restored sampler **skips already-consumed batches**.
- Each batch carries `ids`, `batch_idx`, and `worker_id` alongside the tensors, so the main
  process knows exactly which rows it is consuming.

### 6.3 Queue semantics

With `num_workers = W` and `prefetch_factor = P`, up to `W × P` batches are in flight. Their rows
are `QUEUED` in the DB with their `worker_id` and `batch_idx`. When paused, workers keep
prefetching until the queue is full, then block. The UI shows exactly that full queue.

### 6.4 `TrainingSession`

All model mutation happens on **one training thread**, which reads commands from a
`queue.Queue`. HTTP handlers only enqueue commands and await the result, so there are no locks
around the model.

| Command            | Behaviour |
|--------------------|-----------|
| `step()`           | Pull the next batch from the iterator; forward, loss (per-sample, `reduction="none"`), backward, optimizer step. Then one `Apply` batch for every row in the batch: `Patch{status:CONSUMED, last_loss, last_pred, last_step}` + `Increment{times_seen, 1}`, plus a `session` patch for `global_step`. At the end of an epoch, patch `session.epoch += 1` and create a new iterator. |
| `run(delay_ms)`    | Loop `step()` until `pause()`. Commands in the queue are checked between steps, so Evaluate and Save stay responsive. `delay_ms` (from the UI slider, 0–2000) keeps the queue changes visible. Setting `batch_size=1` gives per-sample granularity. |
| `pause()`          | Stop the run loop after the current step. |
| `evaluate()`       | Save `model.training`, switch to `model.eval()`, and run `torch.no_grad()` forward on `val[val_sample_id]`. Restore the mode afterwards. The training RNG is not touched. Append to `eval_log`, and return `{loss, pred, target, probs, epoch, step}`. |
| `shuffle_val()`    | Pick a uniformly random `val` id ≠ the current one, using a **separate** `random.Random`, and patch `session.val_sample_id`. |
| `select_val(id)`   | Set a specific validation sample. |
| `save(name)`       | Runs between steps. Build the blob with `torch.save({model, optimizer, sampler, rng_states})`, then call `CreateCheckpoint(name, blob)`. |
| `restore(id)`      | Pause, then call `RestoreCheckpoint`, then load the blob. Shut down the old DataLoader. Clear stale `QUEUED` rows for the current epoch (`Apply` patches back to `IDLE`). Recreate the DataLoader from the restored sampler; it re-queues the same batches deterministically. |

**Edits during training (rewrites):** the UI can relabel a row (`Patch y`) or edit it at any time.
Batches already in flight hold the old value. On consume, the main process compares each row's
`fetched_version` with the current `version`. If they differ, the change event is tagged
`stale_consume`, and the UI shows a badge on that row. The edit takes effect on the row's next
fetch.

### 6.5 Graph (`mods.graph`)

- `torch.fx.symbolic_trace(model)`, then `ShapeProp` with one sample from `train` to annotate
  each node with its output shape and dtype. The result is JSON
  `{nodes:[{id, op, target, shape, params}], edges:[[src,dst]]}`.
- If tracing fails (data-dependent control flow), fall back to the `named_modules()` hierarchy as
  a tree, and show a banner in the UI saying so.
- The graph is computed once at startup and is static.

### 6.6 HTTP / WebSocket API (FastAPI, `localhost:8000`)

```
GET  /api/graph                       → graph JSON
GET  /api/state                       → {mode, epoch, step, val_sample_id, loader:{W,P,batch_size}}
GET  /api/table/{name}?after=&limit=&status=   → proxied Scan
POST /api/table/{name}/{id}           body {set:{...}} → Patch (relabel/edit)
POST /api/train/step | /run {delay_ms} | /pause
POST /api/val/evaluate | /shuffle | /select {id}
GET  /api/val/sample                  → current sample (x as nested list, y)
GET  /api/checkpoints                 POST /api/checkpoints {name}
POST /api/checkpoints/{id}/restore    DELETE /api/checkpoints/{id}
GET  /api/db/stats
WS   /ws                              → server-push messages:
     {type:"change", lsn, tag, changes:[...]}      (relayed from Subscribe, filtered to UI columns)
     {type:"step", step, epoch, loss, lr}
     {type:"eval", ...result}
     {type:"reset"}                                 (client re-fetches the queue and table views)
```

The relay batches DB events into one WS frame every 50 ms, so updates reach the browser at about
20 Hz or less, no matter how fast training runs.

## 7. UI

A single page (vanilla JS served by FastAPI; Cytoscape.js + dagre for the graph from jsDelivr).

| Panel          | Contents |
|----------------|----------|
| Model graph    | Top-down DAG: node labels show the op/module and its output shape. |
| Worker queues  | One lane per worker. Each lane lists its in-flight batches in `batch_idx` order, and each batch is a row of sample chips. A chip turns from `QUEUED` to `CONSUMED`, and the consumed batch slides out. The batch that the next Step will consume is highlighted. |
| Controls       | Run / Pause / Step, a speed slider, and a readout of epoch, step, and loss. |
| Validation     | Sample preview (an image for the 8×8 digits), with **Evaluate** and **Shuffle** buttons. Shows prediction, target, class probabilities, and a history of this sample's loss over past evaluations. |
| Table          | A paged view of `train`. Rows changed in the last second flash, and cells can be edited inline (for relabeling). |
| Checkpoints    | List of checkpoints with Save (name) and Restore/Delete buttons. |
| DB panel       | Live `Stats`: last/durable LSN, WAL size and segments, last snapshot LSN, commits/s, group size, p50/p99 commit latency. Also a scrolling log of the latest change events. |

## 8. Example workload

`examples/digits_mlp.py`:

- Load sklearn `load_digits()` (1,797 samples, 8×8). Split 80/20, then `BulkLoad` into `train`
  and `val`.
- Model: `MLP(64 → 64 → 10)` with ReLU; optimizer SGD, lr 0.1. Defaults: `batch_size=16`,
  `num_workers=2`, `prefetch_factor=2`.
- The model can be supplied in two ways: pretrained weights loaded from `--init weights.pt`
  (the precomputed model), or randomly initialized.
- One command starts `modsdb`, waits for its health check, seeds the DB if it is empty, starts
  FastAPI, and opens the browser.

## 9. Testing

**Go (the core of the project):**

- *WAL framing:* round-trip tests. Corrupt one byte of a record, or truncate at every byte
  offset of the last record, and check that recovery yields exactly the records before it.
- *Crash-recovery property test:* apply random op sequences to both the engine and a reference
  `map` model. Kill the engine (skip graceful close) at random points, including mid-snapshot,
  then reopen it. The recovered state must equal the model's state at `durable_lsn`.
- *Snapshot equivalence:* snapshot + replay must equal pure WAL replay from empty.
- *Checkpoint/restore:* restore gives the exact snapshot state, and LSNs keep increasing after it.
  Crashing at each step of `CreateCheckpoint` never leaves a half-made checkpoint visible.
- *Concurrency:* N goroutines issue `Apply` and `FetchAndMark` against one server, under
  `go test -race`. Afterwards there are no lost increments, and `expect_version` conflicts return
  `ABORTED`.
- *Hub:* a slow subscriber gets `LAGGED` and never blocks commits. A resubscribe from an LSN
  still in the ring receives no duplicate events and misses none.
- *Benchmarks:* `Apply` throughput and latency under each fsync policy, and at group sizes
  1/8/64.

**Python:**

- Unit tests against `FakeStore`: sampler determinism, skipping consumed batches, and the
  bookkeeping done by `step()` and `evaluate()`. Also check that `evaluate()` leaves the model's
  mode and the training RNG unchanged.
- *Integration test (real `modsdb` binary):* train 20 steps, save, train 10 more and record their
  batches and losses, restore, train 10 more. The batch ids must match exactly, and the losses
  must match within 1e-6.
- *Queue test:* with W=2 and P=2, while paused, `Count(status=QUEUED)` equals
  `min(4 × batch_size, remaining rows)`.

## 10. Build phases

1. **DB core:** proto, WAL, memtable, commit loop, `Apply`/`GetRows`/`Scan`, and recovery,
   with crash tests.
2. **Snapshots + checkpoints + hub + Stats.**
3. **Python client + DBDataset + TrainingSession** (CLI only; the integration test passes).
4. **FastAPI + WebSocket + UI** (graph, queues, controls, validation, table, checkpoints, DB
   panel).
5. **Optional:** `ReadAt` time travel and the UI scrubber, plus a benchmark page.

## 11. Decisions made on the user's behalf (please confirm)

- **Save state = full session checkpoint** (§4.5). It covers model, optimizer, sampler, RNG, and
  every table. Row history comes from the WAL (time travel in phase 5).
- **"Rewrites"** means both: the platform writes derived fields back to rows (loss, prediction,
  times seen), and the user can edit or relabel rows in the middle of training. Edits apply on
  the row's next fetch, and stale in-flight uses are flagged.
- **No SQLite fallback.** The custom Go DB is the only backend, and `FakeStore` exists only for
  Python unit tests.
- **Local-only:** no authentication, and everything binds to `127.0.0.1`.
