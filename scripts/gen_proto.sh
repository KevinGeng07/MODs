#!/usr/bin/env bash
# Regenerate Go and Python gRPC code from db/proto/modsdb/v1/modsdb.proto.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
export PATH="$(go env GOPATH)/bin:$PATH"

mkdir -p "$ROOT/db/gen/modsdbv1" "$ROOT/python/mods/db/_pb"
"$PY" -m grpc_tools.protoc -I "$ROOT/db/proto" \
  --plugin=protoc-gen-go="$(command -v protoc-gen-go)" \
  --plugin=protoc-gen-go-grpc="$(command -v protoc-gen-go-grpc)" \
  --go_out="$ROOT/db/gen/modsdbv1" --go_opt=paths=source_relative \
  --go-grpc_out="$ROOT/db/gen/modsdbv1" --go-grpc_opt=paths=source_relative \
  modsdb/v1/modsdb.proto
# Flatten modsdb/v1/*.go into db/gen/modsdbv1/
mv "$ROOT"/db/gen/modsdbv1/modsdb/v1/*.go "$ROOT/db/gen/modsdbv1/" && rm -rf "$ROOT/db/gen/modsdbv1/modsdb"

"$PY" -m grpc_tools.protoc -I "$ROOT/db/proto/modsdb/v1" \
  --python_out="$ROOT/python/mods/db/_pb" --pyi_out="$ROOT/python/mods/db/_pb" \
  --grpc_python_out="$ROOT/python/mods/db/_pb" modsdb.proto
# grpc_tools emits an absolute import; make it package-relative.
sed -i '' 's/^import modsdb_pb2 as/from . import modsdb_pb2 as/' "$ROOT/python/mods/db/_pb/modsdb_pb2_grpc.py"
touch "$ROOT/python/mods/db/_pb/__init__.py"
echo "generated"
