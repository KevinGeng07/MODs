// Package server exposes the engine over gRPC.
package server

import (
	"context"
	"errors"

	pb "mods/db/gen/modsdbv1"
	"mods/db/internal/engine"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type Server struct {
	pb.UnimplementedModsDBServer
	e *engine.Engine
}

func New(e *engine.Engine) *Server { return &Server{e: e} }

func toStatus(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, engine.ErrNotFound):
		return status.Error(codes.NotFound, err.Error())
	case errors.Is(err, engine.ErrInvalid):
		return status.Error(codes.InvalidArgument, err.Error())
	case errors.Is(err, engine.ErrClosed):
		return status.Error(codes.Unavailable, err.Error())
	}
	return status.Error(codes.Internal, err.Error())
}

func (s *Server) CreateTable(_ context.Context, r *pb.CreateTableRequest) (*pb.CommitAck, error) {
	lsn, err := s.e.CreateTable(r.Table, r.Schema)
	return &pb.CommitAck{Lsn: lsn}, toStatus(err)
}

func (s *Server) DropTable(_ context.Context, r *pb.DropTableRequest) (*pb.CommitAck, error) {
	lsn, err := s.e.DropTable(r.Table)
	return &pb.CommitAck{Lsn: lsn}, toStatus(err)
}

func (s *Server) GetRows(_ context.Context, r *pb.GetRowsRequest) (*pb.Rows, error) {
	rows, err := s.e.GetRows(r.Table, r.Ids, r.Columns)
	return &pb.Rows{Rows: rows}, toStatus(err)
}

func (s *Server) Scan(_ context.Context, r *pb.ScanRequest) (*pb.Rows, error) {
	rows, err := s.e.Scan(r)
	return &pb.Rows{Rows: rows}, toStatus(err)
}

func (s *Server) Apply(_ context.Context, b *pb.WriteBatch) (*pb.CommitAck, error) {
	lsn, err := s.e.Apply(b)
	return &pb.CommitAck{Lsn: lsn}, toStatus(err)
}

func (s *Server) FetchAndMark(_ context.Context, r *pb.FetchAndMarkRequest) (*pb.Rows, error) {
	rows, err := s.e.FetchAndMark(r)
	return &pb.Rows{Rows: rows}, toStatus(err)
}

func (s *Server) CreateCheckpoint(_ context.Context, r *pb.CreateCheckpointRequest) (*pb.Checkpoint, error) {
	c, err := s.e.CreateCheckpoint(r.Name, r.Blob)
	return c, toStatus(err)
}

func (s *Server) ListCheckpoints(context.Context, *pb.Empty) (*pb.CheckpointList, error) {
	return &pb.CheckpointList{Checkpoints: s.e.ListCheckpoints()}, nil
}

func (s *Server) RestoreCheckpoint(_ context.Context, r *pb.RestoreCheckpointRequest) (*pb.RestoreCheckpointResponse, error) {
	c, blob, lsn, err := s.e.RestoreCheckpoint(r.Id)
	if err != nil {
		return nil, toStatus(err)
	}
	return &pb.RestoreCheckpointResponse{Checkpoint: c, Blob: blob, Lsn: lsn}, nil
}

func (s *Server) Stats(context.Context, *pb.Empty) (*pb.StatsResponse, error) {
	return s.e.Stats(), nil
}
