#실행
chmod +x scripts/gdb_init.sh
scripts/gdb_init.sh


#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-agensgraph}"
DB="${DB_NAME:-agens}"
USER="${DB_USER:-postgres}"

docker exec -i "$CONTAINER_NAME" psql -U "$USER" -d "$DB" < scripts/gdb_init.sql
echo "✅ AgensGraph DDL applied."