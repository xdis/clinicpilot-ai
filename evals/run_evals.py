"""Scenario evaluation for ClinicPilot AI against the running stack (real LLM).

Each case is a scripted multi-turn conversation with per-turn expectations: intent, cited
sources, tools called, reply content, whether a summary awaits confirmation, and whether a
booking was written. Reports turn-level intent accuracy, check pass rate, case success rate,
booking accuracy, latency and token usage, and writes evals/results/latest.md.

    python evals/run_evals.py                        # via the backend gateway (default)
    python evals/run_evals.py --base http://localhost:3000/api

Precondition: start from the seed data (`cd backend && npm run db:demo-reset`), because some
expectations refer to the seeded appointments. Bookings created by the eval are tagged
source=chatbot; run the same command afterwards to remove them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent


def post(base: str, message: str, session: str) -> dict:
    req = urllib.request.Request(
        f"{base}/chat",
        data=json.dumps({"message": message, "sessionId": session, "patientId": "P001"}).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.load(res)


def check_turn(exp: dict, r: dict) -> list[str]:
    fails: list[str] = []
    if "intent" in exp and r["intent"] not in exp["intent"]:
        fails.append(f"intent {r['intent']} not in {exp['intent']}")
    for c in exp.get("cites", []):
        if c not in r["citations"]:
            fails.append(f"missing citation {c} (got {r['citations']})")
    tools = [t["name"] for t in r["tool_calls"]]
    for t in exp.get("tools", []):
        if t not in tools:
            fails.append(f"tool {t} not called")
    reply = r["reply"].lower()
    for s in exp.get("reply_has", []):
        if s.lower() not in reply:
            fails.append(f"reply lacks '{s}'")
    for s in exp.get("reply_lacks", []):
        if s.lower() in reply:
            fails.append(f"reply should not contain '{s}'")
    if "awaiting" in exp and r["awaiting_confirmation"] != exp["awaiting"]:
        fails.append(f"awaiting_confirmation={r['awaiting_confirmation']}, expected {exp['awaiting']}")
    if "booked" in exp and bool(r["booking_created"]) != exp["booked"]:
        fails.append(f"booking_created={bool(r['booking_created'])}, expected {exp['booked']}")
    for key in exp.get("draft_has", []):
        if not (r.get("booking_draft") or {}).get(key):
            fails.append(f"draft missing {key}")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:3000/api")
    ap.add_argument("--only", default=None, help="run a single case id")
    args = ap.parse_args()

    cases = json.loads((HERE / "cases.json").read_text())
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]

    rows, latencies, tokens = [], [], []
    turns_total = intents_ok = checks_total = checks_ok = 0
    booking_turns = booking_ok = 0
    for case in cases:
        session = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
        case_fails: list[str] = []
        for i, exp in enumerate(case["turns"], 1):
            try:
                r = post(args.base, exp["user"], session)
            except urllib.error.URLError as exc:
                case_fails.append(f"turn {i}: request failed: {exc}")
                break
            fails = check_turn(exp, r)
            turns_total += 1
            intents_ok += r["intent"] in exp.get("intent", [r["intent"]])
            n_checks = sum(len(exp.get(k, [])) for k in ("cites", "tools", "reply_has", "reply_lacks", "draft_has")) + sum(
                k in exp for k in ("intent", "awaiting", "booked"))
            checks_total += n_checks
            checks_ok += n_checks - len(fails)
            if "booked" in exp:
                booking_turns += 1
                booking_ok += bool(r["booking_created"]) == exp["booked"]
            latencies.append(r["latency_ms"])
            tokens.append(r["usage"]["input_tokens"] + r["usage"]["output_tokens"])
            case_fails += [f"turn {i}: {f}" for f in fails]
        rows.append((case["id"], len(case["turns"]), case_fails))
        status = "PASS" if not case_fails else "FAIL"
        print(f"{status:4}  {case['id']}" + ("" if not case_fails else "\n      " + "\n      ".join(case_fails)))

    passed = sum(1 for _, _, f in rows if not f)
    p50 = statistics.median(latencies) if latencies else 0
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0
    summary = {
        "cases": f"{passed}/{len(rows)}",
        "intent_accuracy": f"{intents_ok}/{turns_total} ({100 * intents_ok / max(1, turns_total):.0f}%)",
        "checks_passed": f"{checks_ok}/{checks_total} ({100 * checks_ok / max(1, checks_total):.0f}%)",
        "booking_outcome_accuracy": f"{booking_ok}/{booking_turns}",
        "latency_ms_p50": int(p50),
        "latency_ms_p95": int(p95),
        "avg_tokens_per_turn": int(statistics.mean(tokens)) if tokens else 0,
    }
    print("\n" + json.dumps(summary, indent=2))

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    lines = [
        "# ClinicPilot AI - scenario evaluation",
        "",
        f"Run: {time.strftime('%Y-%m-%d %H:%M')} against `{args.base}`",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()],
        "",
        "| Case | Turns | Result |",
        "|---|---|---|",
        *[f"| {cid} | {n} | {'pass' if not f else 'FAIL: ' + '; '.join(f)} |" for cid, n, f in rows],
        "",
    ]
    (out / "latest.md").write_text("\n".join(lines))
    print(f"Report written to {out / 'latest.md'}")


if __name__ == "__main__":
    main()
