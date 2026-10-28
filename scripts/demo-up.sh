#!/usr/bin/env bash
# One command for a live demo: tracing backends (OpenTelemetry Collector + Jaeger, Langfuse) in Docker, then the
# app (backend :3000, AI service :8000, UI :5173) with tracing on and a small short-term budget so the rolling
# summary shows up within a few turns.  Ctrl+C stops the app; the tracing containers keep running.
#   ./scripts/demo-up.sh            then, in another terminal:  ./scripts/demo-check.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

docker info >/dev/null 2>&1 || { echo "Docker is not running. Open Docker Desktop (if it misbehaves: ~/luff/projects/aitools/tec/docker-doctor/docker-doctor.sh)." >&2; exit 1; }
for p in 3000 8000 5173; do
  if lsof -iTCP:"$p" -sTCP:LISTEN -P >/dev/null 2>&1; then
    echo "Port $p is in use by:" >&2; lsof -iTCP:"$p" -sTCP:LISTEN -P | tail -n +2 >&2
    echo "Stop it first (e.g. an older ClinicPilot run), then start again." >&2; exit 1
  fi
done

echo "Starting tracing backends..."
docker compose -f observability/otel/docker-compose.yml up -d >/dev/null
docker compose -f observability/langfuse/docker-compose.yml -f observability/langfuse/docker-compose.override.yml up -d >/dev/null
printf "Waiting for Langfuse"
for _ in $(seq 1 60); do curl -sf localhost:3100/api/public/health >/dev/null && break; printf "."; sleep 3; done; echo

(cd backend && npx prisma migrate deploy >/dev/null)   # apply new migrations to the local database

export OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4318}"
export SHORT_TERM_MAX_TOKENS="${SHORT_TERM_MAX_TOKENS:-400}"
echo "Langfuse http://localhost:3100 (login in observability/langfuse/.env) · Jaeger http://localhost:16686"
exec ./scripts/dev.sh
