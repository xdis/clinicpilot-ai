#!/usr/bin/env bash
# Starts backend (3000), AI service (8000) and frontend (5173). Ctrl+C stops all three.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/.logs"

if ! grep -qE '^LLM_PROVIDER=fake' "$ROOT/ai-service/.env" 2>/dev/null \
  && { ! grep -qE '^LLM_API_KEY=.+' "$ROOT/ai-service/.env" 2>/dev/null || grep -qE '^LLM_API_KEY=your-api-key-here' "$ROOT/ai-service/.env"; }; then
  echo "ai-service/.env has no LLM_API_KEY. Add one (or set LLM_PROVIDER=fake for an offline smoke test)." >&2
  exit 1
fi

pids=()
cleanup() { echo; echo "Stopping..."; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

(cd "$ROOT/backend" && npm run start:dev > "$ROOT/.logs/backend.log" 2>&1) & pids+=($!)
(cd "$ROOT/ai-service" && .venv/bin/uvicorn app.main:app --port 8000 > "$ROOT/.logs/ai-service.log" 2>&1) & pids+=($!)
(cd "$ROOT/frontend" && npx vite --port 5173 > "$ROOT/.logs/frontend.log" 2>&1) & pids+=($!)

printf "Waiting for services"
for _ in $(seq 1 90); do
  if curl -sf localhost:3000/api/health >/dev/null && curl -sf localhost:8000/health >/dev/null && curl -sf localhost:5173 >/dev/null; then
    echo
    echo "Ready:"
    echo "  UI          http://localhost:5173"
    echo "  Backend API http://localhost:3000/api   (Swagger: /api/docs)"
    echo "  AI service  http://localhost:8000/health"
    echo "Logs in .logs/. Press Ctrl+C to stop."
    wait
    exit 0
  fi
  printf "."; sleep 1
done
echo; echo "Services did not become healthy in time; see .logs/*.log" >&2
exit 1
