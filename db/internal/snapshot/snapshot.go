// Package snapshot writes and reads full point-in-time copies of every table.
//
// File layout (little-endian):
//
//	"MODSNAP1" | format u32 | lsn u64 | ntables u32
//	per table:  len u32 | CreateTableRequest proto | nrows u64 | (len u32 | Row proto)*
//	crc u32    (CRC32C of every preceding byte)
//
// Files are written to a temp name, fsynced, renamed, and the directory
// fsynced, so a snapshot is either fully present or absent.
package snapshot

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"

	pb "mods/db/gen/modsdbv1"

	"google.golang.org/protobuf/proto"
)

const (
	magic         = "MODSNAP1"
	formatVersion = 1
	maxMsg        = 256 << 20
)

var ErrBadSnapshot = errors.New("snapshot: corrupt or truncated")

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// Table is one table's schema and rows. Rows are shared, never mutated.
type Table struct {
	Name   string
	Schema *pb.Schema
	Rows   []*pb.Row
}

// Path returns the canonical file path for a snapshot at lsn.
func Path(dir string, lsn uint64) string {
	return filepath.Join(dir, fmt.Sprintf("%020d.snap", lsn))
}

type hashWriter struct {
	w *bufio.Writer
	h hash.Hash32
	n int64
}

func (hw *hashWriter) Write(p []byte) (int, error) {
	hw.h.Write(p)
	hw.n += int64(len(p))
	return hw.w.Write(p)
}

func (hw *hashWriter) u32(v uint32) error {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	_, err := hw.Write(b[:])
	return err
}

func (hw *hashWriter) u64(v uint64) error {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	_, err := hw.Write(b[:])
	return err
}

func (hw *hashWriter) msg(m proto.Message) error {
	b, err := proto.Marshal(m)
	if err != nil {
		return err
	}
	if err := hw.u32(uint32(len(b))); err != nil {
		return err
	}
	_, err = hw.Write(b)
	return err
}

// Write atomically writes a snapshot to path and returns its size in bytes.
func Write(path string, lsn uint64, tables []Table) (int64, error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return 0, err
	}
	final := path
	tmp := final + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return 0, err
	}
	ok := false
	defer func() {
		if !ok {
			f.Close()
			os.Remove(tmp)
		}
	}()
	hw := &hashWriter{w: bufio.NewWriterSize(f, 1<<20), h: crc32.New(castagnoli)}
	if _, err := hw.Write([]byte(magic)); err != nil {
		return 0, err
	}
	hw.u32(formatVersion)
	hw.u64(lsn)
	hw.u32(uint32(len(tables)))
	for _, t := range tables {
		if err := hw.msg(&pb.CreateTableRequest{Table: t.Name, Schema: t.Schema}); err != nil {
			return 0, err
		}
		hw.u64(uint64(len(t.Rows)))
		for _, r := range t.Rows {
			if err := hw.msg(r); err != nil {
				return 0, err
			}
		}
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], hw.h.Sum32())
	if _, err := hw.w.Write(crc[:]); err != nil {
		return 0, err
	}
	if err := hw.w.Flush(); err != nil {
		return 0, err
	}
	if err := f.Sync(); err != nil {
		return 0, err
	}
	if err := f.Close(); err != nil {
		return 0, err
	}
	if err := os.Rename(tmp, final); err != nil {
		return 0, err
	}
	ok = true
	if err := syncDir(dir); err != nil {
		return 0, err
	}
	return hw.n + 4, nil
}

type hashReader struct {
	r *bufio.Reader
	h hash.Hash32
}

func (hr *hashReader) full(p []byte) error {
	if _, err := io.ReadFull(hr.r, p); err != nil {
		return ErrBadSnapshot
	}
	hr.h.Write(p)
	return nil
}

func (hr *hashReader) u32() (uint32, error) {
	var b [4]byte
	err := hr.full(b[:])
	return binary.LittleEndian.Uint32(b[:]), err
}

func (hr *hashReader) u64() (uint64, error) {
	var b [8]byte
	err := hr.full(b[:])
	return binary.LittleEndian.Uint64(b[:]), err
}

func (hr *hashReader) msg(m proto.Message) error {
	n, err := hr.u32()
	if err != nil {
		return err
	}
	if n > maxMsg {
		return ErrBadSnapshot
	}
	b := make([]byte, n)
	if err := hr.full(b); err != nil {
		return err
	}
	if err := proto.Unmarshal(b, m); err != nil {
		return ErrBadSnapshot
	}
	return nil
}

// Read loads and verifies a snapshot file.
func Read(path string) (uint64, []Table, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, nil, err
	}
	defer f.Close()
	hr := &hashReader{r: bufio.NewReaderSize(f, 1<<20), h: crc32.New(castagnoli)}
	head := make([]byte, len(magic))
	if err := hr.full(head); err != nil || string(head) != magic {
		return 0, nil, ErrBadSnapshot
	}
	if v, err := hr.u32(); err != nil || v != formatVersion {
		return 0, nil, ErrBadSnapshot
	}
	lsn, err := hr.u64()
	if err != nil {
		return 0, nil, err
	}
	nt, err := hr.u32()
	if err != nil {
		return 0, nil, err
	}
	tables := make([]Table, 0, nt)
	for i := uint32(0); i < nt; i++ {
		var info pb.CreateTableRequest
		if err := hr.msg(&info); err != nil {
			return 0, nil, err
		}
		nr, err := hr.u64()
		if err != nil {
			return 0, nil, err
		}
		t := Table{Name: info.Table, Schema: info.Schema, Rows: make([]*pb.Row, 0, nr)}
		for j := uint64(0); j < nr; j++ {
			r := &pb.Row{}
			if err := hr.msg(r); err != nil {
				return 0, nil, err
			}
			t.Rows = append(t.Rows, r)
		}
		tables = append(tables, t)
	}
	want := hr.h.Sum32()
	var crc [4]byte
	if _, err := io.ReadFull(hr.r, crc[:]); err != nil || binary.LittleEndian.Uint32(crc[:]) != want {
		return 0, nil, ErrBadSnapshot
	}
	return lsn, tables, nil
}

func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
