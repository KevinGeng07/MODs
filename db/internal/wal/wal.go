// Package wal implements the write-ahead log: framed, checksummed records in
// size-rotated segment files named by the LSN of their first record.
//
// Frame layout (little-endian):
//
//	len u32 | crc u32 | lsn u64 | type u8 | payload[len]
//	crc = CRC32C(lsn ‖ type ‖ payload)
package wal

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Record types.
const (
	TypeCreateTable uint8 = 1
	TypeDropTable   uint8 = 2
	TypeWriteBatch  uint8 = 3
)

const (
	headerSize     = 17
	maxPayload     = 256 << 20
	segmentSuffix  = ".log"
	DefaultSegment = 64 << 20
)

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// ErrCorrupt reports a damaged record before the final segment.
var ErrCorrupt = errors.New("wal: corrupt record in non-final segment")

type Record struct {
	LSN     uint64
	Type    uint8
	Payload []byte
}

func checksum(lsn uint64, typ uint8, payload []byte) uint32 {
	var b [9]byte
	binary.LittleEndian.PutUint64(b[:8], lsn)
	b[8] = typ
	c := crc32.Update(0, castagnoli, b[:])
	return crc32.Update(c, castagnoli, payload)
}

// EncodeTo writes one framed record and returns the number of bytes written.
func EncodeTo(w io.Writer, r Record) (int, error) {
	var h [headerSize]byte
	binary.LittleEndian.PutUint32(h[0:4], uint32(len(r.Payload)))
	binary.LittleEndian.PutUint32(h[4:8], checksum(r.LSN, r.Type, r.Payload))
	binary.LittleEndian.PutUint64(h[8:16], r.LSN)
	h[16] = r.Type
	if _, err := w.Write(h[:]); err != nil {
		return 0, err
	}
	if _, err := w.Write(r.Payload); err != nil {
		return 0, err
	}
	return headerSize + len(r.Payload), nil
}

// decode reads one record. ok=false with err=nil means a torn or corrupt
// record (or clean EOF when n==0 bytes were available).
func decode(r *bufio.Reader) (rec Record, size int, ok bool, err error) {
	var h [headerSize]byte
	n, err := io.ReadFull(r, h[:])
	if err == io.EOF {
		return rec, 0, false, io.EOF
	}
	if err != nil { // short header
		return rec, n, false, nil
	}
	plen := binary.LittleEndian.Uint32(h[0:4])
	if plen > maxPayload {
		return rec, n, false, nil
	}
	payload := make([]byte, plen)
	if _, err := io.ReadFull(r, payload); err != nil {
		return rec, n, false, nil
	}
	rec = Record{LSN: binary.LittleEndian.Uint64(h[8:16]), Type: h[16], Payload: payload}
	if checksum(rec.LSN, rec.Type, rec.Payload) != binary.LittleEndian.Uint32(h[4:8]) {
		return rec, n, false, nil
	}
	return rec, headerSize + int(plen), true, nil
}

type segment struct {
	first uint64
	path  string
}

func segmentName(first uint64) string { return fmt.Sprintf("%020d%s", first, segmentSuffix) }

func listSegments(dir string) ([]segment, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var segs []segment
	for _, e := range ents {
		name := e.Name()
		if !strings.HasSuffix(name, segmentSuffix) {
			continue
		}
		first, err := strconv.ParseUint(strings.TrimSuffix(name, segmentSuffix), 10, 64)
		if err != nil {
			continue
		}
		segs = append(segs, segment{first: first, path: filepath.Join(dir, name)})
	}
	sort.Slice(segs, func(i, j int) bool { return segs[i].first < segs[j].first })
	return segs, nil
}

// ReplayResult summarises a Replay call.
type ReplayResult struct {
	LastLSN       uint64 // highest valid LSN seen (0 if none)
	TruncatedTail bool   // a torn tail was cut off
}

// Replay calls fn for every valid record with LSN > from, in order. A damaged
// record in the last segment is treated as a torn write (a crash mid-append)
// and the file is truncated there. Damage in an earlier segment cannot be a
// torn write, so it returns ErrCorrupt.
func Replay(dir string, from uint64, fn func(Record) error) (ReplayResult, error) {
	var res ReplayResult
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return res, err
	}
	segs, err := listSegments(dir)
	if err != nil {
		return res, err
	}
	for i, s := range segs {
		// Skip whole segments that end at or before `from`.
		if i+1 < len(segs) && segs[i+1].first <= from+1 {
			continue
		}
		cutAt, err := replaySegment(s, from, &res, fn)
		if err != nil {
			return res, err
		}
		if cutAt < 0 {
			continue
		}
		if i != len(segs)-1 {
			return res, fmt.Errorf("%w: %s at offset %d", ErrCorrupt, filepath.Base(s.path), cutAt)
		}
		if err := os.Truncate(s.path, cutAt); err != nil {
			return res, err
		}
		res.TruncatedTail = true
		return res, syncDir(dir)
	}
	return res, nil
}

// replaySegment returns the byte offset of the first bad record, or -1.
func replaySegment(s segment, from uint64, res *ReplayResult, fn func(Record) error) (int64, error) {
	f, err := os.Open(s.path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)
	var off int64
	for {
		rec, size, ok, err := decode(r)
		if err == io.EOF {
			return -1, nil
		}
		if !ok || (res.LastLSN != 0 && rec.LSN <= res.LastLSN) {
			return off, nil
		}
		off += int64(size)
		res.LastLSN = rec.LSN
		if rec.LSN <= from {
			continue
		}
		if err := fn(rec); err != nil {
			return 0, err
		}
	}
}

// Writer appends records to the active segment. It is not safe for concurrent
// use; the engine's single commit goroutine owns it.
type Writer struct {
	dir        string
	maxSegment int64
	f          *os.File
	buf        *bufio.Writer
	active     segment
	activeSize int64
}

// OpenWriter starts a fresh segment whose first record will be nextLSN.
func OpenWriter(dir string, nextLSN uint64, maxSegment int64) (*Writer, error) {
	if maxSegment <= 0 {
		maxSegment = DefaultSegment
	}
	w := &Writer{dir: dir, maxSegment: maxSegment}
	if err := w.startSegment(nextLSN); err != nil {
		return nil, err
	}
	return w, nil
}

func (w *Writer) startSegment(first uint64) error {
	path := filepath.Join(w.dir, segmentName(first))
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return err
	}
	// A leftover file with this name (an empty segment from a previous run) is reused.
	w.f, w.buf = f, bufio.NewWriterSize(f, 1<<20)
	w.active, w.activeSize = segment{first: first, path: path}, st.Size()
	return syncDir(w.dir)
}

// Append buffers a record. Call Sync to make it durable.
func (w *Writer) Append(r Record) error {
	if w.activeSize >= w.maxSegment {
		if err := w.rotate(r.LSN); err != nil {
			return err
		}
	}
	n, err := EncodeTo(w.buf, r)
	w.activeSize += int64(n)
	return err
}

func (w *Writer) rotate(nextLSN uint64) error {
	if err := w.Sync(); err != nil {
		return err
	}
	if err := w.f.Close(); err != nil {
		return err
	}
	return w.startSegment(nextLSN)
}

// Sync flushes and fsyncs the active segment.
func (w *Writer) Sync() error {
	if err := w.buf.Flush(); err != nil {
		return err
	}
	return w.f.Sync()
}

// TruncateThrough deletes closed segments whose records all have LSN <= lsn.
func (w *Writer) TruncateThrough(lsn uint64) error {
	segs, err := listSegments(w.dir)
	if err != nil {
		return err
	}
	removed := false
	for i, s := range segs {
		if s.path == w.active.path || i+1 >= len(segs) || segs[i+1].first > lsn+1 {
			continue
		}
		if err := os.Remove(s.path); err != nil {
			return err
		}
		removed = true
	}
	if removed {
		return syncDir(w.dir)
	}
	return nil
}

func (w *Writer) Close() error {
	if err := w.Sync(); err != nil {
		return err
	}
	return w.f.Close()
}

func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
