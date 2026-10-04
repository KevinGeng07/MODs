package snapshot

import (
	"errors"
	"os"
	"testing"

	pb "mods/db/gen/modsdbv1"

	"google.golang.org/protobuf/proto"
)

func sample() []Table {
	schema := &pb.Schema{Columns: []*pb.Column{{Name: "y", Type: pb.ColumnType_INT64}}}
	var rows []*pb.Row
	for i := uint64(1); i <= 50; i++ {
		rows = append(rows, &pb.Row{Id: i, Version: i, Cols: map[string]*pb.Value{"y": {V: &pb.Value_I{I: int64(i)}}}})
	}
	return []Table{{Name: "train", Schema: schema, Rows: rows}, {Name: "empty", Schema: schema}}
}

func TestRoundTrip(t *testing.T) {
	dir := t.TempDir()
	in := sample()
	n, err := Write(Path(dir, 42), 42, in)
	if err != nil || n <= 0 {
		t.Fatalf("write: %d %v", n, err)
	}
	if st, _ := os.Stat(Path(dir, 42)); st.Size() != n {
		t.Fatalf("size %d != reported %d", st.Size(), n)
	}
	lsn, out, err := Read(Path(dir, 42))
	if err != nil || lsn != 42 || len(out) != 2 {
		t.Fatalf("read: lsn=%d tables=%d err=%v", lsn, len(out), err)
	}
	for i := range in[0].Rows {
		if !proto.Equal(in[0].Rows[i], out[0].Rows[i]) {
			t.Fatalf("row %d differs", i)
		}
	}
}

func TestCorruptionDetected(t *testing.T) {
	dir := t.TempDir()
	Write(Path(dir, 7), 7, sample())
	p := Path(dir, 7)
	b, _ := os.ReadFile(p)
	for _, mutate := range []func([]byte) []byte{
		func(b []byte) []byte { c := append([]byte{}, b...); c[len(c)/2] ^= 1; return c },
		func(b []byte) []byte { return b[:len(b)-1] },
		func(b []byte) []byte { return b[:20] },
	} {
		os.WriteFile(p, mutate(b), 0o644)
		if _, _, err := Read(p); !errors.Is(err, ErrBadSnapshot) {
			t.Fatalf("want ErrBadSnapshot, got %v", err)
		}
	}
}
