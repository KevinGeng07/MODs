package wal

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func writeN(t *testing.T, dir string, first, n uint64, maxSeg int64) {
	t.Helper()
	w, err := OpenWriter(dir, first, maxSeg)
	if err != nil {
		t.Fatal(err)
	}
	for i := first; i < first+n; i++ {
		if err := w.Append(Record{LSN: i, Type: TypeWriteBatch, Payload: []byte(fmt.Sprintf("rec-%d", i))}); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
}

func replayAll(t *testing.T, dir string, from uint64) ([]Record, ReplayResult) {
	t.Helper()
	var got []Record
	res, err := Replay(dir, from, false, func(r Record) error { got = append(got, r); return nil })
	if err != nil {
		t.Fatal(err)
	}
	return got, res
}

func TestRoundTrip(t *testing.T) {
	dir := t.TempDir()
	writeN(t, dir, 1, 100, 0)
	got, res := replayAll(t, dir, 0)
	if len(got) != 100 || res.LastLSN != 100 || res.TruncatedTail {
		t.Fatalf("got %d records, res %+v", len(got), res)
	}
	for i, r := range got {
		if r.LSN != uint64(i+1) || string(r.Payload) != fmt.Sprintf("rec-%d", i+1) {
			t.Fatalf("record %d = %+v", i, r)
		}
	}
	got, _ = replayAll(t, dir, 60)
	if len(got) != 40 || got[0].LSN != 61 {
		t.Fatalf("from=60: %d records, first %d", len(got), got[0].LSN)
	}
}

func TestRotationAndTruncate(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWriter(dir, 1, 200) // tiny segments
	if err != nil {
		t.Fatal(err)
	}
	for i := uint64(1); i <= 50; i++ {
		if err := w.Append(Record{LSN: i, Type: TypeWriteBatch, Payload: bytes.Repeat([]byte{'x'}, 30)}); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Sync(); err != nil {
		t.Fatal(err)
	}
	if w.Segments() < 5 {
		t.Fatalf("expected rotation, got %d segments", w.Segments())
	}
	before := w.Size()
	if err := w.TruncateThrough(30); err != nil {
		t.Fatal(err)
	}
	if w.Size() >= before {
		t.Fatalf("size did not shrink: %d -> %d", before, w.Size())
	}
	w.Close()
	got, _ := replayAll(t, dir, 30)
	if len(got) != 20 || got[0].LSN != 31 {
		t.Fatalf("after truncate: %d records, first %v", len(got), got)
	}
}

// Truncating the last record at every byte offset must recover exactly the
// preceding records and cut the file back to a clean boundary.
func TestTornTailEveryOffset(t *testing.T) {
	base := t.TempDir()
	writeN(t, base, 1, 5, 0)
	path := filepath.Join(base, segmentName(1))
	full, _ := os.ReadFile(path)
	recLen := headerSize + len("rec-5")
	cleanLen := len(full) - recLen
	for cut := cleanLen; cut < len(full); cut++ {
		dir := t.TempDir()
		p := filepath.Join(dir, segmentName(1))
		os.WriteFile(p, full[:cut], 0o644)
		got, res := replayAll(t, dir, 0)
		if len(got) != 4 || res.LastLSN != 4 {
			t.Fatalf("cut=%d: got %d records", cut, len(got))
		}
		if cut > cleanLen && !res.TruncatedTail {
			t.Fatalf("cut=%d: expected truncation", cut)
		}
		st, _ := os.Stat(p)
		if st.Size() != int64(cleanLen) {
			t.Fatalf("cut=%d: file size %d want %d", cut, st.Size(), cleanLen)
		}
	}
}

func TestCorruptByteInLastSegment(t *testing.T) {
	dir := t.TempDir()
	writeN(t, dir, 1, 10, 0)
	path := filepath.Join(dir, segmentName(1))
	b, _ := os.ReadFile(path)
	recLen := headerSize + len("rec-1")
	b[3*recLen+headerSize] ^= 0xff // payload byte of record 4
	os.WriteFile(path, b, 0o644)
	got, res := replayAll(t, dir, 0)
	if len(got) != 3 || !res.TruncatedTail {
		t.Fatalf("got %d records, res %+v", len(got), res)
	}
}

func TestCorruptEarlierSegmentRefuses(t *testing.T) {
	dir := t.TempDir()
	writeN(t, dir, 1, 10, 0)
	writeN(t, dir, 11, 10, 0)
	path := filepath.Join(dir, segmentName(1))
	b, _ := os.ReadFile(path)
	b[headerSize] ^= 0xff
	os.WriteFile(path, b, 0o644)
	_, err := Replay(dir, 0, false, func(Record) error { return nil })
	if !errors.Is(err, ErrCorrupt) {
		t.Fatalf("want ErrCorrupt, got %v", err)
	}
	var n int
	res, err := Replay(dir, 0, true, func(Record) error { n++; return nil })
	if err != nil || n != 0 || !res.TruncatedTail {
		t.Fatalf("force: n=%d err=%v res=%+v", n, err, res)
	}
	if segs, _ := listSegments(dir); len(segs) != 1 {
		t.Fatalf("later segments not removed: %d", len(segs))
	}
}
