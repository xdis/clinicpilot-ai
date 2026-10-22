#!/usr/bin/env bash
# Installs dependencies for all three services and prepares the database. Safe to re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"

echo "==> backend (NestJS + Prisma)"
cd "$ROOT/backend"
[ -f .env ] || cp .env.example .env
npm install
npx prisma generate
npm run db:setup

echo "==> ai-service (Python)"
cd "$ROOT/ai-service"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-dev.txt
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    Created ai-service/.env - put your LLM_API_KEY in it before starting."
fi

echo "==> frontend (React + Vite)"
cd "$ROOT/frontend"
npm install

echo "Setup complete. Start everything with: ./scripts/dev.sh"
