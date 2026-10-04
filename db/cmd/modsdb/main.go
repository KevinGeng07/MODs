// Command modsdb runs the MODs database as a local gRPC service.
package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	pb "mods/db/gen/modsdbv1"
	"mods/db/internal/engine"
	"mods/db/internal/server"

	"google.golang.org/grpc"
)

func main() {
	dir := flag.String("data-dir", "./modsdb-data", "directory for WAL, snapshots and checkpoints")
	addr := flag.String("addr", "127.0.0.1:7070", "listen address")
	snapEvery := flag.Int64("snapshot-every", 16<<20, "WAL bytes between automatic snapshots")
	flag.Parse()

	e, err := engine.Open(engine.Options{Dir: *dir, SnapshotEvery: *snapEvery})
	if err != nil {
		log.Fatalf("modsdb: recovery failed: %v", err)
	}
	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatal(err)
	}
	gs := grpc.NewServer(grpc.MaxRecvMsgSize(256<<20), grpc.MaxSendMsgSize(256<<20))
	pb.RegisterModsDBServer(gs, server.New(e))

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	go func() { <-sig; gs.Stop() }()

	fmt.Printf("modsdb listening on %s (lsn %d)\n", lis.Addr(), e.Stats().LastLsn) // the launcher waits for this line
	if err := gs.Serve(lis); err != nil {
		log.Printf("modsdb: %v", err)
	}
	if err := e.Close(); err != nil {
		log.Fatalf("modsdb: close: %v", err)
	}
}
