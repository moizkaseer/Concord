#!/usr/bin/env bash
# Reproduce the scorecard from a fresh clone: start Redis in Docker, run the
# fault-injection scenarios and the kill-and-resume trials, print the scorecard.
# Arguments go to the harness, for example --check or --seeds 4.
set -euo pipefail
cd "$(dirname "$0")/.."

need() {
  command -v "$1" >/dev/null || { echo "error: $1 is not installed. $2" >&2; exit 1; }
}
need uv "Install it from https://docs.astral.sh/uv/getting-started/installation/"
need docker "Install it from https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1 || { echo "error: Docker is installed but not running." >&2; exit 1; }

port="${CONCORD_REDIS_PORT:-6379}"
echo "Starting Redis on port $port..."
if ! CONCORD_REDIS_PORT="$port" docker compose up --detach --wait redis; then
  echo "error: Redis did not start. If port $port is taken, pick another:" >&2
  echo "  CONCORD_REDIS_PORT=6380 $0" >&2
  exit 1
fi

# Database 14 keeps the harness's keys apart from anything else in that Redis.
export REDIS_URL="redis://localhost:$port/14"
uv run python -m concord.harness "$@"
