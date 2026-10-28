#!/usr/bin/env bash
# Builds the three images, loads them into a local kind cluster and deploys ClinicPilot.
#   ./scripts/kind-up.sh            -> http://localhost:8088
# Needs: docker, kind, kubectl. Tracing backends (optional) run next to the cluster:
#   docker compose -f observability/otel/docker-compose.yml up -d
#   docker compose -f observability/langfuse/docker-compose.yml -f observability/langfuse/docker-compose.override.yml up -d
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=clinicpilot
cd "$ROOT"

kind get clusters | grep -qx "$CLUSTER" || kind create cluster --config k8s/kind-config.yaml

docker build -q -t clinicpilot/backend:dev -f backend/Dockerfile .
docker build -q -t clinicpilot/ai-service:dev ai-service
docker build -q -t clinicpilot/frontend:dev frontend
kind load docker-image --name "$CLUSTER" clinicpilot/backend:dev clinicpilot/ai-service:dev clinicpilot/frontend:dev

kubectl apply -f k8s/clinicpilot.yaml >/dev/null   # creates the namespace first
# secrets from ai-service/.env (never committed); empty Langfuse keys simply leave Langfuse off
val() { grep -E "^$1=" ai-service/.env | head -1 | cut -d= -f2- || true; }
kubectl -n clinicpilot create secret generic clinicpilot-secrets \
  --from-literal=LLM_API_KEY="$(val LLM_API_KEY)" \
  --from-literal=LANGFUSE_PUBLIC_KEY="$(val LANGFUSE_PUBLIC_KEY)" \
  --from-literal=LANGFUSE_SECRET_KEY="$(val LANGFUSE_SECRET_KEY)" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

kubectl -n clinicpilot rollout restart deploy >/dev/null   # pick up rebuilt :dev images
for d in backend ai-service frontend; do kubectl -n clinicpilot rollout status deploy/$d --timeout=240s; done
kubectl -n clinicpilot get pods -o wide
echo "ClinicPilot on kind: http://localhost:8088"
