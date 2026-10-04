// Package engine is modsdb's storage engine: an in-memory table image made
// durable by a write-ahead log and periodic snapshots, plus named checkpoints.
//
// All writes go through one commit goroutine. It drains every queued request,
// validates each against the committed state plus earlier requests in the same
// group, appends them to the WAL, fsyncs once (group commit), and only then
// applies them to memory and acknowledges. Readers never see non-durable data.
package engine

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	pb "mods/db/gen/modsdbv1"
	"mods/db/internal/snapshot"
	"mods/db/internal/wal"

	"google.golang.org/protobuf/proto"
)

var (
	ErrNotFound = errors.New("not found")
	ErrInvalid  = errors.New("invalid argument")
	ErrClosed   = errors.New("database is closed")
)

type Options struct {
	Dir           string
	SnapshotEvery int64 // WAL bytes between automatic snapshots (default 16 MiB)
	SegmentSize   int64 // WAL segment size (default 64 MiB)
}

type table struct {
	schema *pb.Schema
	types  map[string]pb.ColumnType
	rows   map[uint64]*pb.Row // rows are immutable; writes swap pointers
}

func newTable(s *pb.Schema) *table {
	t := &table{schema: s, types: map[string]pb.ColumnType{}, rows: map[uint64]*pb.Row{}}
	for _, c := range s.GetColumns() {
		t.types[c.Name] = c.Type
	}
	return t
}

type reqKind int

const (
	reqWrite reqKind = iota
	reqFetch
	reqCreate
	reqDrop
	reqCheckpoint
	reqRestore
	reqClose
)

type request struct {
	kind   reqKind
	batch  *pb.WriteBatch
	fetch  *pb.FetchAndMarkRequest
	create *pb.CreateTableRequest
	name   string
	blob   []byte
	id     string
	done   chan result
}

type result struct {
	lsn  uint64
	rows []*pb.Row
	ckpt *pb.Checkpoint
	blob []byte
	err  error
}

type Engine struct {
	opts                     Options
	walDir, snapDir, ckptDir string

	mu     sync.RWMutex // guards tables (the loop is the only writer)
	tables map[string]*table

	w         *wal.Writer // commit loop only
	lastLSN   atomic.Uint64
	lastSnap  atomic.Uint64
	sinceSnap int64 // WAL bytes since the last snapshot (commit loop only)
	commits   atomic.Uint64
	fsyncs    atomic.Uint64
	failed    error // set on WAL failure; the engine then refuses writes (commit loop only)

	reqs    chan *request
	stopped chan struct{}
	closed  atomic.Bool
}

// Open recovers the database in opts.Dir (latest readable snapshot + WAL
// replay) and starts the commit loop.
func Open(opts Options) (*Engine, error) {
	if opts.SnapshotEvery <= 0 {
		opts.SnapshotEvery = 16 << 20
	}
	e := &Engine{opts: opts, tables: map[string]*table{}, reqs: make(chan *request, 1024), stopped: make(chan struct{}),
		walDir: filepath.Join(opts.Dir, "wal"), snapDir: filepath.Join(opts.Dir, "snap"), ckptDir: filepath.Join(opts.Dir, "ckpt")}
	for _, d := range []string{e.walDir, e.snapDir, e.ckptDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return nil, err
		}
	}
	if err := e.recover(); err != nil {
		return nil, err
	}
	w, err := wal.OpenWriter(e.walDir, e.lastLSN.Load()+1, opts.SegmentSize)
	if err != nil {
		return nil, err
	}
	e.w = w
	go e.loop()
	return e, nil
}

// snapshotLSNs lists snapshot files in dir, newest first.
func snapshotLSNs(dir string) []uint64 {
	ents, _ := os.ReadDir(dir)
	var out []uint64
	for _, ent := range ents {
		name := ent.Name()
		if strings.HasSuffix(name, ".tmp") {
			os.Remove(filepath.Join(dir, name)) // interrupted write
			continue
		}
		if lsn, err := strconv.ParseUint(strings.TrimSuffix(name, ".snap"), 10, 64); err == nil {
			out = append(out, lsn)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] > out[j] })
	return out
}

func (e *Engine) load(tables []snapshot.Table) {
	e.tables = map[string]*table{}
	for _, st := range tables {
		t := newTable(st.Schema)
		for _, r := range st.Rows {
			t.rows[r.Id] = r
		}
		e.tables[st.Name] = t
	}
}

func (e *Engine) recover() error {
	var base uint64
	for _, lsn := range snapshotLSNs(e.snapDir) {
		if _, tables, err := snapshot.Read(snapshot.Path(e.snapDir, lsn)); err == nil {
			e.load(tables)
			base = lsn
			break
		}
	}
	res, err := wal.Replay(e.walDir, base, false, func(r wal.Record) error {
		switch r.Type {
		case wal.TypeCreateTable:
			var m pb.CreateTableRequest
			if err := proto.Unmarshal(r.Payload, &m); err != nil {
				return err
			}
			e.tables[m.Table] = newTable(m.Schema)
		case wal.TypeDropTable:
			var m pb.DropTableRequest
			if err := proto.Unmarshal(r.Payload, &m); err != nil {
				return err
			}
			delete(e.tables, m.Table)
		case wal.TypeWriteBatch:
			var m pb.WriteBatch
			if err := proto.Unmarshal(r.Payload, &m); err != nil {
				return err
			}
			e.applyLogged(&m, r.LSN)
		case wal.TypeRestore: // crashed between logging a restore and snapshotting it
			var m pb.RestoreCheckpointRequest
			if err := proto.Unmarshal(r.Payload, &m); err != nil {
				return err
			}
			_, tables, err := snapshot.Read(filepath.Join(e.ckptDir, m.Id+".snap"))
			if err != nil {
				return fmt.Errorf("replaying restore of %s: %w", m.Id, err)
			}
			e.load(tables)
		default:
			return fmt.Errorf("unknown WAL record type %d", r.Type)
		}
		return nil
	})
	if err != nil {
		return err
	}
	e.lastLSN.Store(max(base, res.LastLSN))
	e.lastSnap.Store(base)
	return nil
}

// applyLogged applies a WAL-form batch (PUT = full row, PATCH = post-values).
// It is the only mutation path, used both live and during recovery.
func (e *Engine) applyLogged(b *pb.WriteBatch, lsn uint64) {
	for _, op := range b.Ops {
		t := e.tables[op.Table]
		if t == nil {
			continue
		}
		cols := op.Cols
		if op.Kind == pb.Op_PATCH {
			old := t.rows[op.Id]
			if old == nil {
				continue
			}
			cols = merge(old.Cols, op.Cols)
		}
		t.rows[op.Id] = &pb.Row{Id: op.Id, Version: lsn, Cols: cols}
	}
}

func merge(base, set map[string]*pb.Value) map[string]*pb.Value {
	out := make(map[string]*pb.Value, len(base)+len(set))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range set {
		out[k] = v
	}
	return out
}

// ------------------------------------------------------------ validation

type key struct {
	table string
	id    uint64
}

// staging holds rows written by validated-but-unapplied requests.
type staging struct {
	parent *staging
	rows   map[key]*pb.Row
}

func (e *Engine) get(s *staging, tbl string, id uint64) *pb.Row {
	for ; s != nil; s = s.parent {
		if r, ok := s.rows[key{tbl, id}]; ok {
			return r
		}
	}
	return e.tables[tbl].rows[id]
}

func checkCols(t *table, cols map[string]*pb.Value) error {
	for name, v := range cols {
		typ, ok := t.types[name]
		if !ok {
			return fmt.Errorf("%w: unknown column %q", ErrInvalid, name)
		}
		switch x := v.GetV().(type) {
		case *pb.Value_I:
			ok = typ == pb.ColumnType_INT64
		case *pb.Value_F:
			ok = typ == pb.ColumnType_FLOAT64
		case *pb.Value_T:
			n := 1
			for _, d := range x.T.Shape {
				n *= int(d)
			}
			size := map[pb.DType]int{pb.DType_F32: 4, pb.DType_U8: 1}[x.T.Dtype]
			ok = typ == pb.ColumnType_TENSOR && size > 0 && n*size == len(x.T.Data)
		default:
			ok = false
		}
		if !ok {
			return fmt.Errorf("%w: bad value for column %q", ErrInvalid, name)
		}
	}
	return nil
}

// resolve validates a batch and returns its WAL form, staging the new rows.
func (e *Engine) resolve(s *staging, b *pb.WriteBatch, lsn uint64) (*pb.WriteBatch, error) {
	out := &pb.WriteBatch{Tag: b.Tag}
	for _, op := range b.Ops {
		t := e.tables[op.Table]
		if t == nil {
			return nil, fmt.Errorf("%w: table %q", ErrNotFound, op.Table)
		}
		if err := checkCols(t, op.Cols); err != nil {
			return nil, err
		}
		cur := e.get(s, op.Table, op.Id)
		if op.Kind != pb.Op_PUT && cur == nil {
			return nil, fmt.Errorf("%w: %s/%d", ErrNotFound, op.Table, op.Id)
		}
		logged := &pb.Op{Table: op.Table, Id: op.Id, Kind: op.Kind, Cols: op.Cols}
		var cols map[string]*pb.Value
		switch op.Kind {
		case pb.Op_PUT:
			cols = op.Cols
		case pb.Op_PATCH:
			cols = merge(cur.Cols, op.Cols)
		case pb.Op_INCR:
			set := map[string]*pb.Value{}
			for c, d := range op.Cols {
				switch dv := d.V.(type) {
				case *pb.Value_I:
					set[c] = &pb.Value{V: &pb.Value_I{I: cur.Cols[c].GetI() + dv.I}}
				case *pb.Value_F:
					set[c] = &pb.Value{V: &pb.Value_F{F: cur.Cols[c].GetF() + dv.F}}
				default:
					return nil, fmt.Errorf("%w: INCR of non-numeric column %q", ErrInvalid, c)
				}
			}
			// Logged as a PATCH of the resulting values, so replay is idempotent.
			logged = &pb.Op{Table: op.Table, Id: op.Id, Kind: pb.Op_PATCH, Cols: set}
			cols = merge(cur.Cols, set)
		default:
			return nil, fmt.Errorf("%w: op kind", ErrInvalid)
		}
		s.rows[key{op.Table, op.Id}] = &pb.Row{Id: op.Id, Version: lsn, Cols: cols}
		out.Ops = append(out.Ops, logged)
	}
	return out, nil
}

func project(r *pb.Row, columns []string) *pb.Row {
	if len(columns) == 0 {
		return r
	}
	out := &pb.Row{Id: r.Id, Version: r.Version, Cols: map[string]*pb.Value{}}
	for _, c := range columns {
		if v, ok := r.Cols[c]; ok {
			out.Cols[c] = v
		}
	}
	return out
}

// ------------------------------------------------------------ commit loop

func (e *Engine) loop() {
	defer close(e.stopped)
	for req := range e.reqs {
		group := []*request{req}
	drain:
		for len(group) < 512 {
			select {
			case r := <-e.reqs:
				group = append(group, r)
			default:
				break drain
			}
		}
		// Writes batch together; anything else is a barrier handled alone.
		var writes []*request
		for _, r := range group {
			if r.kind == reqWrite || r.kind == reqFetch {
				writes = append(writes, r)
				continue
			}
			e.commit(writes)
			writes = nil
			if e.barrier(r) {
				return
			}
		}
		e.commit(writes)
		if e.sinceSnap >= e.opts.SnapshotEvery && e.failed == nil {
			if err := e.snapshot(e.lastLSN.Load()); err != nil {
				e.failed = err
			}
		}
	}
}

func (e *Engine) logRecord(typ uint8, m proto.Message) (uint64, error) {
	payload, err := proto.Marshal(m)
	if err != nil {
		return 0, err
	}
	lsn := e.lastLSN.Load() + 1
	if err := e.w.Append(wal.Record{LSN: lsn, Type: typ, Payload: payload}); err != nil {
		return 0, err
	}
	e.sinceSnap += int64(len(payload)) + 17
	return lsn, nil
}

type pendingWrite struct {
	req    *request
	lsn    uint64
	logged *pb.WriteBatch
	rows   []*pb.Row // FetchAndMark result
}

func (e *Engine) commit(reqs []*request) {
	if len(reqs) == 0 {
		return
	}
	if e.failed != nil {
		for _, r := range reqs {
			r.done <- result{err: e.failed}
		}
		return
	}
	group := &staging{rows: map[key]*pb.Row{}}
	var ok []pendingWrite
	for _, r := range reqs {
		s := &staging{parent: group, rows: map[key]*pb.Row{}}
		lsn := e.lastLSN.Load() + 1
		batch, rows := r.batch, []*pb.Row(nil)
		if r.kind == reqFetch {
			// Read each row (pre-mark image), then patch the mark onto it.
			if e.tables[r.fetch.Table] == nil {
				r.done <- result{err: fmt.Errorf("%w: table %q", ErrNotFound, r.fetch.Table)}
				continue
			}
			batch = &pb.WriteBatch{Tag: r.fetch.Tag}
			for _, id := range r.fetch.Ids {
				if cur := e.get(group, r.fetch.Table, id); cur != nil {
					rows = append(rows, project(cur, r.fetch.Columns))
					batch.Ops = append(batch.Ops, &pb.Op{Table: r.fetch.Table, Id: id, Kind: pb.Op_PATCH, Cols: r.fetch.Mark})
				}
			}
		}
		logged, err := e.resolve(s, batch, lsn)
		if err == nil && len(logged.Ops) == 0 {
			r.done <- result{lsn: e.lastLSN.Load(), rows: rows}
			continue
		}
		if err == nil {
			_, err = e.logRecord(wal.TypeWriteBatch, logged)
		}
		if err != nil {
			r.done <- result{err: err}
			continue
		}
		e.lastLSN.Store(lsn)
		for k, v := range s.rows {
			group.rows[k] = v
		}
		ok = append(ok, pendingWrite{req: r, lsn: lsn, logged: logged, rows: rows})
	}
	if len(ok) == 0 {
		return
	}
	if err := e.w.Sync(); err != nil {
		e.failed = fmt.Errorf("WAL fsync failed, database is read-only: %w", err)
		for _, p := range ok {
			p.req.done <- result{err: e.failed}
		}
		return
	}
	e.fsyncs.Add(1)
	e.mu.Lock()
	for _, p := range ok {
		e.applyLogged(p.logged, p.lsn)
	}
	e.mu.Unlock()
	for _, p := range ok {
		e.commits.Add(1)
		p.req.done <- result{lsn: p.lsn, rows: p.rows}
	}
}

func (e *Engine) capture() []snapshot.Table {
	names := make([]string, 0, len(e.tables))
	for n := range e.tables {
		names = append(names, n)
	}
	sort.Strings(names)
	var out []snapshot.Table
	for _, n := range names {
		t := e.tables[n]
		rows := make([]*pb.Row, 0, len(t.rows))
		for _, r := range t.rows {
			rows = append(rows, r)
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].Id < rows[j].Id })
		out = append(out, snapshot.Table{Name: n, Schema: t.schema, Rows: rows})
	}
	return out
}

// snapshot writes the full image at lsn, keeps the two newest snapshots and
// deletes WAL segments that neither of them needs.
func (e *Engine) snapshot(lsn uint64) error {
	if _, err := snapshot.Write(snapshot.Path(e.snapDir, lsn), lsn, e.capture()); err != nil {
		return err
	}
	e.sinceSnap = 0
	e.lastSnap.Store(lsn)
	snaps := snapshotLSNs(e.snapDir)
	for _, old := range snaps[min(2, len(snaps)):] {
		os.Remove(snapshot.Path(e.snapDir, old))
	}
	return e.w.TruncateThrough(snaps[min(1, len(snaps)-1)])
}

func newID() string {
	var b [6]byte
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// barrier runs one non-write request. Returns true when the loop should stop.
func (e *Engine) barrier(r *request) bool {
	if e.failed != nil && r.kind != reqClose {
		r.done <- result{err: e.failed}
		return false
	}
	switch r.kind {
	case reqCreate:
		if _, exists := e.tables[r.create.Table]; exists {
			r.done <- result{lsn: e.lastLSN.Load()} // idempotent
			return false
		}
		lsn, err := e.logRecord(wal.TypeCreateTable, r.create)
		if err == nil {
			err = e.w.Sync()
		}
		if err != nil {
			r.done <- result{err: err}
			return false
		}
		e.lastLSN.Store(lsn)
		e.mu.Lock()
		e.tables[r.create.Table] = newTable(r.create.Schema)
		e.mu.Unlock()
		r.done <- result{lsn: lsn}

	case reqDrop:
		if _, exists := e.tables[r.create.Table]; !exists {
			r.done <- result{err: fmt.Errorf("%w: table %q", ErrNotFound, r.create.Table)}
			return false
		}
		lsn, err := e.logRecord(wal.TypeDropTable, &pb.DropTableRequest{Table: r.create.Table})
		if err == nil {
			err = e.w.Sync()
		}
		if err != nil {
			r.done <- result{err: err}
			return false
		}
		e.lastLSN.Store(lsn)
		e.mu.Lock()
		delete(e.tables, r.create.Table)
		e.mu.Unlock()
		r.done <- result{lsn: lsn}

	case reqCheckpoint:
		// The loop is the only writer, so the image captured here is consistent.
		c := &pb.Checkpoint{Id: newID(), Name: r.name, Lsn: e.lastLSN.Load(), CreatedUnixMs: time.Now().UnixMilli()}
		err := e.writeCheckpoint(c, r.blob)
		r.done <- result{ckpt: c, err: err}

	case reqRestore:
		c, blob, tables, err := e.readCheckpoint(r.id)
		if err != nil {
			r.done <- result{err: err}
			return false
		}
		lsn, err := e.logRecord(wal.TypeRestore, &pb.RestoreCheckpointRequest{Id: r.id})
		if err == nil {
			err = e.w.Sync()
		}
		if err != nil {
			r.done <- result{err: err}
			return false
		}
		e.lastLSN.Store(lsn)
		e.mu.Lock()
		e.load(tables)
		e.mu.Unlock()
		if err := e.snapshot(lsn); err != nil {
			e.failed = err
		}
		r.done <- result{lsn: lsn, ckpt: c, blob: blob, err: err}

	case reqClose:
		var err error
		if e.failed == nil && e.lastLSN.Load() > e.lastSnap.Load() {
			err = e.snapshot(e.lastLSN.Load())
		}
		if cerr := e.w.Close(); err == nil {
			err = cerr
		}
		r.done <- result{err: err}
		return true
	}
	return false
}

// ------------------------------------------------------------ checkpoints
// A checkpoint is ckpt/<id>.snap + <id>.bin + <id>.json. The JSON metadata is
// written last, so a checkpoint exists only once all three are durable.

func writeAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err = f.Write(data); err == nil {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (e *Engine) writeCheckpoint(c *pb.Checkpoint, blob []byte) error {
	base := filepath.Join(e.ckptDir, c.Id)
	if _, err := snapshot.Write(base+".snap", c.Lsn, e.capture()); err != nil {
		return err
	}
	if err := writeAtomic(base+".bin", blob); err != nil {
		return err
	}
	meta, _ := json.Marshal(map[string]any{"id": c.Id, "name": c.Name, "lsn": c.Lsn, "created_unix_ms": c.CreatedUnixMs})
	return writeAtomic(base+".json", meta)
}

func (e *Engine) readMeta(id string) (*pb.Checkpoint, error) {
	b, err := os.ReadFile(filepath.Join(e.ckptDir, id+".json"))
	if err != nil {
		return nil, fmt.Errorf("%w: checkpoint %q", ErrNotFound, id)
	}
	var m struct {
		ID      string `json:"id"`
		Name    string `json:"name"`
		LSN     uint64 `json:"lsn"`
		Created int64  `json:"created_unix_ms"`
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	return &pb.Checkpoint{Id: m.ID, Name: m.Name, Lsn: m.LSN, CreatedUnixMs: m.Created}, nil
}

func (e *Engine) readCheckpoint(id string) (*pb.Checkpoint, []byte, []snapshot.Table, error) {
	c, err := e.readMeta(id)
	if err != nil {
		return nil, nil, nil, err
	}
	blob, err := os.ReadFile(filepath.Join(e.ckptDir, id+".bin"))
	if err != nil {
		return nil, nil, nil, err
	}
	_, tables, err := snapshot.Read(filepath.Join(e.ckptDir, id+".snap"))
	return c, blob, tables, err
}

// ------------------------------------------------------------ public API

func (e *Engine) submit(r *request) result {
	if e.closed.Load() {
		return result{err: ErrClosed}
	}
	r.done = make(chan result, 1)
	e.reqs <- r
	return <-r.done
}

func (e *Engine) CreateTable(name string, s *pb.Schema) (uint64, error) {
	if name == "" || len(s.GetColumns()) == 0 {
		return 0, fmt.Errorf("%w: table needs a name and columns", ErrInvalid)
	}
	res := e.submit(&request{kind: reqCreate, create: &pb.CreateTableRequest{Table: name, Schema: s}})
	return res.lsn, res.err
}

// DropTable deletes a table and all its rows (durably, through the WAL).
func (e *Engine) DropTable(name string) (uint64, error) {
	res := e.submit(&request{kind: reqDrop, create: &pb.CreateTableRequest{Table: name}})
	return res.lsn, res.err
}

func (e *Engine) Apply(b *pb.WriteBatch) (uint64, error) {
	res := e.submit(&request{kind: reqWrite, batch: b})
	return res.lsn, res.err
}

// FetchAndMark returns the rows (as they were before the mark) and patches
// mark onto each of them in one atomic, durable write.
func (e *Engine) FetchAndMark(r *pb.FetchAndMarkRequest) ([]*pb.Row, error) {
	res := e.submit(&request{kind: reqFetch, fetch: r})
	return res.rows, res.err
}

func (e *Engine) GetRows(tbl string, ids []uint64, columns []string) ([]*pb.Row, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	t := e.tables[tbl]
	if t == nil {
		return nil, fmt.Errorf("%w: table %q", ErrNotFound, tbl)
	}
	var out []*pb.Row
	for _, id := range ids {
		if r := t.rows[id]; r != nil {
			out = append(out, project(r, columns))
		}
	}
	return out, nil
}

func (e *Engine) Scan(r *pb.ScanRequest) ([]*pb.Row, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	t := e.tables[r.Table]
	if t == nil {
		return nil, fmt.Errorf("%w: table %q", ErrNotFound, r.Table)
	}
	var out []*pb.Row
	for _, row := range t.rows {
		if r.Filter && (row.Cols["status"].GetI() != r.Status || row.Cols["epoch"].GetI() != r.Epoch) {
			continue
		}
		out = append(out, project(row, r.Columns))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Id < out[j].Id })
	return out, nil
}

func (e *Engine) CreateCheckpoint(name string, blob []byte) (*pb.Checkpoint, error) {
	res := e.submit(&request{kind: reqCheckpoint, name: name, blob: blob})
	return res.ckpt, res.err
}

func (e *Engine) ListCheckpoints() []*pb.Checkpoint {
	ents, _ := os.ReadDir(e.ckptDir)
	var out []*pb.Checkpoint
	for _, ent := range ents {
		if id, ok := strings.CutSuffix(ent.Name(), ".json"); ok {
			if c, err := e.readMeta(id); err == nil {
				out = append(out, c)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Lsn < out[j].Lsn })
	return out
}

// RestoreCheckpoint replaces every table with the checkpoint's image and
// returns its blob and the LSN of the restore.
func (e *Engine) RestoreCheckpoint(id string) (*pb.Checkpoint, []byte, uint64, error) {
	res := e.submit(&request{kind: reqRestore, id: id})
	return res.ckpt, res.blob, res.lsn, res.err
}

func (e *Engine) Stats() *pb.StatsResponse {
	s := &pb.StatsResponse{LastLsn: e.lastLSN.Load(), LastSnapshotLsn: e.lastSnap.Load(),
		Commits: e.commits.Load(), Fsyncs: e.fsyncs.Load()}
	ents, _ := os.ReadDir(e.walDir)
	for _, ent := range ents {
		if info, err := ent.Info(); err == nil {
			s.WalBytes += uint64(info.Size())
			s.WalSegments++
		}
	}
	return s
}

// Close writes a final snapshot and stops the commit loop.
func (e *Engine) Close() error {
	if e.closed.Swap(true) {
		return nil
	}
	r := &request{kind: reqClose, done: make(chan result, 1)}
	e.reqs <- r
	res := <-r.done
	<-e.stopped
	return res.err
}
