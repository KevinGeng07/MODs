package engine

import (
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"testing"

	pb "mods/db/gen/modsdbv1"
)

func I(v int64) *pb.Value { return &pb.Value{V: &pb.Value_I{I: v}} }

var schema = &pb.Schema{Columns: []*pb.Column{
	{Name: "y", Type: pb.ColumnType_INT64},
	{Name: "status", Type: pb.ColumnType_INT64},
	{Name: "epoch", Type: pb.ColumnType_INT64},
	{Name: "seen", Type: pb.ColumnType_INT64},
}}

func open(t *testing.T, dir string, snapEvery int64) *Engine {
	t.Helper()
	e, err := Open(Options{Dir: dir, SnapshotEvery: snapEvery, SegmentSize: 4 << 10})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func op(kind pb.Op_Kind, id uint64, cols map[string]*pb.Value) *pb.Op {
	return &pb.Op{Table: "t", Id: id, Kind: kind, Cols: cols}
}

func apply(t *testing.T, e *Engine, ops ...*pb.Op) uint64 {
	t.Helper()
	lsn, err := e.Apply(&pb.WriteBatch{Ops: ops})
	if err != nil {
		t.Fatal(err)
	}
	return lsn
}

func dump(t *testing.T, e *Engine) map[uint64]map[string]int64 {
	rows, err := e.Scan(&pb.ScanRequest{Table: "t"})
	if err != nil {
		t.Fatal(err)
	}
	out := map[uint64]map[string]int64{}
	for _, r := range rows {
		out[r.Id] = map[string]int64{}
		for k, v := range r.Cols {
			out[r.Id][k] = v.GetI()
		}
	}
	return out
}

// crashImage copies the data dir while the engine is idle: what a process
// crash right now would leave. WAL is copied before snapshots so a concurrent
// truncation can never remove records a copied snapshot does not cover.
func crashImage(t *testing.T, dir string) string {
	dst := t.TempDir()
	for _, sub := range []string{"wal", "snap", "ckpt"} {
		os.MkdirAll(filepath.Join(dst, sub), 0o755)
		ents, _ := os.ReadDir(filepath.Join(dir, sub))
		for _, ent := range ents {
			in, err := os.Open(filepath.Join(dir, sub, ent.Name()))
			if err != nil {
				continue // removed while copying
			}
			out, _ := os.Create(filepath.Join(dst, sub, ent.Name()))
			io.Copy(out, in)
			in.Close()
			out.Close()
		}
	}
	return dst
}

func TestApplyFetchScan(t *testing.T) {
	e := open(t, t.TempDir(), 0)
	defer e.Close()
	if _, err := e.CreateTable("t", schema); err != nil {
		t.Fatal(err)
	}
	apply(t, e, op(pb.Op_PUT, 1, map[string]*pb.Value{"y": I(3)}), op(pb.Op_PUT, 2, map[string]*pb.Value{"y": I(4), "status": I(0)}))
	lsn := apply(t, e, op(pb.Op_INCR, 1, map[string]*pb.Value{"seen": I(2)}), op(pb.Op_INCR, 1, map[string]*pb.Value{"seen": I(1)}))
	rows, _ := e.GetRows("t", []uint64{1}, nil)
	if rows[0].Version != lsn || rows[0].Cols["seen"].GetI() != 3 {
		t.Fatalf("row = %v", rows[0])
	}

	got, err := e.FetchAndMark(&pb.FetchAndMarkRequest{Table: "t", Ids: []uint64{2, 99},
		Mark: map[string]*pb.Value{"status": I(1), "epoch": I(5)}, Columns: []string{"y", "status"}})
	if err != nil || len(got) != 1 || got[0].Cols["status"].GetI() != 0 || len(got[0].Cols) != 2 {
		t.Fatalf("fetch = %v %v", got, err)
	}
	queued, _ := e.Scan(&pb.ScanRequest{Table: "t", Filter: true, Status: 1, Epoch: 5})
	if len(queued) != 1 || queued[0].Id != 2 {
		t.Fatalf("queued = %v", queued)
	}

	// Errors, and a failing op aborts its whole batch.
	for _, b := range []*pb.WriteBatch{
		{Ops: []*pb.Op{op(pb.Op_PATCH, 42, map[string]*pb.Value{"y": I(1)})}},
		{Ops: []*pb.Op{op(pb.Op_PATCH, 1, map[string]*pb.Value{"nope": I(1)})}},
		{Ops: []*pb.Op{op(pb.Op_PATCH, 1, map[string]*pb.Value{"y": I(9)}), op(pb.Op_PATCH, 42, nil)}},
	} {
		if _, err := e.Apply(b); !errors.Is(err, ErrNotFound) && !errors.Is(err, ErrInvalid) {
			t.Fatalf("want error, got %v", err)
		}
	}
	if dump(t, e)[1]["y"] != 3 {
		t.Fatal("partial batch was applied")
	}
}

// Random ops against a reference map; at random points recover a crash image
// and compare. Tiny thresholds force segment rotation, snapshots and WAL
// truncation throughout.
func TestCrashRecoveryProperty(t *testing.T) {
	dir := t.TempDir()
	e := open(t, dir, 4<<10)
	defer e.Close()
	e.CreateTable("t", schema)
	rng := rand.New(rand.NewSource(1))
	ref := map[uint64]map[string]int64{}
	for step := 0; step < 1500; step++ {
		id := uint64(rng.Intn(30) + 1)
		if _, ok := ref[id]; !ok {
			y := rng.Int63n(10)
			apply(t, e, op(pb.Op_PUT, id, map[string]*pb.Value{"y": I(y)}))
			ref[id] = map[string]int64{"y": y}
		} else if rng.Intn(2) == 0 {
			apply(t, e, op(pb.Op_INCR, id, map[string]*pb.Value{"seen": I(1)}))
			ref[id]["seen"]++
		} else {
			s := rng.Int63n(3)
			apply(t, e, op(pb.Op_PATCH, id, map[string]*pb.Value{"status": I(s)}))
			ref[id]["status"] = s
		}
		if rng.Intn(150) == 0 {
			rec := open(t, crashImage(t, dir), 4<<10)
			got := dump(t, rec)
			for id, cols := range ref {
				for c, v := range cols {
					if got[id][c] != v {
						t.Fatalf("step %d: row %d %s = %d, want %d", step, id, c, got[id][c], v)
					}
				}
			}
			if len(got) != len(ref) || rec.Stats().LastLsn != e.Stats().LastLsn {
				t.Fatalf("step %d: %d rows lsn %d, want %d rows lsn %d", step, len(got), rec.Stats().LastLsn, len(ref), e.Stats().LastLsn)
			}
			rec.Close()
		}
	}
	if e.Stats().LastSnapshotLsn == 0 {
		t.Fatal("no snapshot was taken")
	}
}

func TestCheckpointRestore(t *testing.T) {
	dir := t.TempDir()
	e := open(t, dir, 0)
	e.CreateTable("t", schema)
	apply(t, e, op(pb.Op_PUT, 1, map[string]*pb.Value{"y": I(1)}))
	want := dump(t, e)
	c, err := e.CreateCheckpoint("c", []byte("weights"))
	if err != nil {
		t.Fatal(err)
	}
	apply(t, e, op(pb.Op_INCR, 1, map[string]*pb.Value{"seen": I(10)}), op(pb.Op_PUT, 2, nil))
	before := e.Stats().LastLsn
	_, blob, lsn, err := e.RestoreCheckpoint(c.Id)
	if err != nil || string(blob) != "weights" || lsn <= before {
		t.Fatalf("restore: blob=%q lsn=%d err=%v", blob, lsn, err)
	}
	if fmt.Sprint(dump(t, e)) != fmt.Sprint(want) {
		t.Fatalf("after restore %v, want %v", dump(t, e), want)
	}
	next := apply(t, e, op(pb.Op_INCR, 1, map[string]*pb.Value{"seen": I(1)}))
	after := dump(t, e)
	img := crashImage(t, dir)
	e.Close()
	for _, d := range []string{img, dir} {
		r := open(t, d, 0)
		if fmt.Sprint(dump(t, r)) != fmt.Sprint(after) || r.Stats().LastLsn != next || len(r.ListCheckpoints()) != 1 {
			t.Fatalf("reopen %s: %v lsn %d", d, dump(t, r), r.Stats().LastLsn)
		}
		r.Close()
	}
	if _, _, _, err := open(t, dir, 0).RestoreCheckpoint("nope"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing checkpoint: %v", err)
	}
}

func TestConcurrentWritersGroupCommit(t *testing.T) {
	e := open(t, t.TempDir(), 0)
	defer e.Close()
	e.CreateTable("t", schema)
	apply(t, e, op(pb.Op_PUT, 1, nil))
	var wg sync.WaitGroup
	for w := 0; w < 16; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				if _, err := e.Apply(&pb.WriteBatch{Ops: []*pb.Op{op(pb.Op_INCR, 1, map[string]*pb.Value{"seen": I(1)})}}); err != nil {
					t.Error(err)
				}
				e.GetRows("t", []uint64{1}, nil)
			}
		}()
	}
	wg.Wait()
	st := e.Stats()
	if got := dump(t, e)[1]["seen"]; got != 800 {
		t.Fatalf("lost increments: %d", got)
	}
	if st.Fsyncs >= st.Commits {
		t.Logf("no grouping observed: %d commits, %d fsyncs", st.Commits, st.Fsyncs)
	}
}

func TestUint8TensorsAndDropTable(t *testing.T) {
	dir := t.TempDir()
	e := open(t, dir, 0)
	img := &pb.Schema{Columns: []*pb.Column{{Name: "x", Type: pb.ColumnType_TENSOR}}}
	e.CreateTable("imgs", img)
	e.CreateTable("t", schema)
	u8 := &pb.Value{V: &pb.Value_T{T: &pb.Tensor{Shape: []int64{2, 2}, Data: []byte{0, 1, 2, 255}, Dtype: pb.DType_U8}}}
	if _, err := e.Apply(&pb.WriteBatch{Ops: []*pb.Op{{Table: "imgs", Id: 1, Kind: pb.Op_PUT, Cols: map[string]*pb.Value{"x": u8}}}}); err != nil {
		t.Fatal(err)
	}
	// 4 bytes is right for a 2x2 uint8 tensor but wrong for float32.
	f32 := &pb.Value{V: &pb.Value_T{T: &pb.Tensor{Shape: []int64{2, 2}, Data: []byte{0, 1, 2, 255}}}}
	if _, err := e.Apply(&pb.WriteBatch{Ops: []*pb.Op{{Table: "imgs", Id: 2, Kind: pb.Op_PUT, Cols: map[string]*pb.Value{"x": f32}}}}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("float32 size check: %v", err)
	}
	apply(t, e, op(pb.Op_PUT, 1, map[string]*pb.Value{"y": I(1)}))
	if _, err := e.DropTable("t"); err != nil {
		t.Fatal(err)
	}
	if _, err := e.Scan(&pb.ScanRequest{Table: "t"}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("dropped table still readable: %v", err)
	}
	if _, err := e.DropTable("t"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("double drop: %v", err)
	}
	// Both the drop and the uint8 row survive a crash image and a clean reopen.
	img2 := crashImage(t, dir)
	e.Close()
	for _, d := range []string{img2, dir} {
		r := open(t, d, 0)
		rows, err := r.GetRows("imgs", []uint64{1}, nil)
		if err != nil || len(rows) != 1 || rows[0].Cols["x"].GetT().Dtype != pb.DType_U8 {
			t.Fatalf("%s: uint8 row = %v %v", d, rows, err)
		}
		if _, err := r.Scan(&pb.ScanRequest{Table: "t"}); !errors.Is(err, ErrNotFound) {
			t.Fatalf("%s: drop not durable: %v", d, err)
		}
		r.Close()
	}
}
