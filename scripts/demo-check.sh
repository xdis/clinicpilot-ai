#!/usr/bin/env bash
# Pre-flight check before a demo: every service, the LLM key, tracing end to end, voice, memory, conflicts.
# Leaves the demo data clean (demo bookings removed, P001's memory erased).   ./scripts/demo-check.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API=${API:-http://localhost:3000/api}; AI=${AI:-http://localhost:8000}; UI=${UI:-http://localhost:5173}
pass=0; fail=0
ok()  { printf "  \033[32mPASS\033[0m  %s\n" "$*"; pass=$((pass+1)); }
bad() { printf "  \033[31mFAIL\033[0m  %s\n" "$*"; fail=$((fail+1)); }
j() { python3 -c "import json,sys;d=json.load(sys.stdin);print($1)" 2>/dev/null; }
env_get() { grep -E "^$1=" "$2" 2>/dev/null | head -1 | cut -d= -f2-; }

echo "Services"
curl -sf "$UI" >/dev/null && ok "UI $UI" || bad "UI $UI not reachable"
curl -sf "$API/health" >/dev/null && ok "backend $API" || bad "backend $API not reachable"
H=$(curl -sf "$AI/health"); [ -n "$H" ] && ok "AI service: $(echo "$H" | j 'd["llm"], "memory="+str(d["memory"]), "tracing="+str(d["tracing"]), "stt="+str(d["speech_to_text"])')" || bad "AI service $AI not reachable"
curl -sf localhost:3100/api/public/health >/dev/null && ok "Langfuse http://localhost:3100" || bad "Langfuse not running (./scripts/demo-up.sh starts it)"
curl -sf localhost:16686/ >/dev/null && ok "Jaeger http://localhost:16686" || bad "Jaeger not running"

echo "LLM key"
KEY=$(env_get LLM_API_KEY "$ROOT/ai-service/.env")
code=$(curl -s -o /dev/null -w '%{http_code}' https://api.deepseek.com/models -H "Authorization: Bearer $KEY")
[ "$code" = 200 ] && ok "DeepSeek key valid" || bad "DeepSeek key rejected (HTTP $code): put a new key in ai-service/.env"

echo "End-to-end"
(cd "$ROOT/backend" && npm run db:demo-reset >/dev/null 2>&1)
curl -s -X DELETE "$API/patients/P001/memory" >/dev/null
R=$(curl -s -X POST "$API/chat" -H 'content-type: application/json' -d '{"message":"What are the opening hours of Metro Care?","sessionId":"preflight","patientId":"P001"}')
TID=$(echo "$R" | j 'd.get("trace_id") or ""')
echo "$R" | j 'd["intent"]' | grep -q clinic_info && ok "chat answers (intent clinic_info, cites $(echo "$R" | j '",".join(d["citations"])'))" || bad "chat failed: $(echo "$R" | head -c 200)"
if [ -n "$TID" ]; then
  sleep 10
  svc=$(curl -s "localhost:16686/api/traces/$TID" | j 'sorted({p["serviceName"] for t in d["data"] for p in t["processes"].values()})')
  echo "$svc" | grep -q backend && echo "$svc" | grep -q ai-service && ok "Jaeger trace $TID spans both services" || bad "Jaeger trace missing services: $svc"
  PK=$(env_get LANGFUSE_PUBLIC_KEY "$ROOT/ai-service/.env"); SK=$(env_get LANGFUSE_SECRET_KEY "$ROOT/ai-service/.env")
  n=$(curl -s -u "$PK:$SK" "localhost:3100/api/public/v2/observations?traceId=$TID&limit=100" | j 'len(d["data"])')
  [ "${n:-0}" -gt 5 ] && ok "Langfuse has $n observations for the same trace id" || bad "Langfuse observations: ${n:-none}"
else
  bad "no trace_id in the chat response (Langfuse keys missing in ai-service/.env?)"
fi
C=$(curl -s -X POST "$API/chat" -H 'content-type: application/json' -d '{"message":"Book a general health checkup with Dr Jenkins on November 5 at 9am.","sessionId":"preflight-2","patientId":"P001"}')
echo "$C" | grep -q "already have this appointment" && ok "slot conflict detected" || bad "conflict check: $(echo "$C" | j 'd["reply"][:120]')"
A=$(curl -s -X POST "$API/summarize/audio" -F "file=@$ROOT/mock-data/audio/consultation-sample.mp3;type=audio/mpeg")
echo "$A" | j 'd["summary"]["diagnosis"]' | grep -qi pharyngitis && ok "voice summary ($(echo "$A" | j 'str(d["transcribe_ms"])+" ms transcription"'))" || bad "voice summary: $(echo "$A" | head -c 200)"
M=$(curl -s "$API/patients/P001/memory"); echo "$M" | j 'd["enabled"]' | grep -q True && ok "memory API" || bad "memory API: $M"

(cd "$ROOT/backend" && npm run db:demo-reset >/dev/null 2>&1); curl -s -X DELETE "$API/patients/P001/memory" >/dev/null
echo; echo "$pass passed, $fail failed  (demo data reset)"
[ "$fail" = 0 ]
