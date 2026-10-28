"""Agent evaluation for ClinicPilot: quality, consistency, performance and cost, against the running stack.

Layers (each one answers a different question):
  1. Outcome checks (code)     did it do the right thing?   intent, citations, tools, booking written, reply content
                                                              (the same checks as evals/run_evals.py)
  2. Trajectory (agentevals)   did it take the right path?  graph route matches the intent's contract (strict match),
                                                              expected tool calls present (superset match)
  3. Groundedness (judge)      is the answer supported?     openevals RAG_GROUNDEDNESS_PROMPT, judge = a separate LLM
                                                              call; context = the database records the answer cites
  4. Memory (LongMemEval)      does memory help, safely?    information extraction, multi-session, knowledge update,
                                                              temporal, abstention, + healthcare safety (evals/memory_cases.json)
  5. Consistency               does it work every time?    --repeat k: pass@1 (mean) and pass^k (all k runs pass, tau-bench)
  6. Performance and cost      fast and cheap enough?       latency p50/p95/max, tokens, LLM calls, cost per turn
  7. Regression gates          is it better than before?    compare with a baseline run, thresholds in evals/gates.json

Results: evals/results/<run>.json + latest.md. With LANGFUSE_* set, the run is also a Langfuse experiment on the
dataset "clinicpilot-agent-eval" (scores per item, comparable across runs in the UI).

    ai-service/.venv/bin/python evals/agent_eval.py --base http://localhost:3000/api --repeat 3 --judge
    ai-service/.venv/bin/python evals/agent_eval.py --suite memory
    ai-service/.venv/bin/python evals/agent_eval.py --baseline evals/results/baseline.json   # exit 1 on regression

Isolation: every case starts from the seed bookings (`npm run db:demo-reset`, removes bookings made by earlier
cases and repetitions; skip with --no-reset when the backend is not local) and from an empty long-term memory for
P001 (DELETE /patients/P001/memory), so cases cannot influence each other; memory cases wait (poll) for background
writes before the next session.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_evals import check_turn  # noqa: E402  (same outcome checks as the original runner)

PATIENT = "P001"

# The graph's routing contract (ai-service/app/agent.py _route): which nodes a turn must pass for its intent.
ROUTES = {
    "booking": ["load_context", "guard", "analyze", "booking"],
    "booking_query": ["load_context", "guard", "analyze", "lookup_bookings", "answer"],
    "emergency": [["load_context", "guard", "emergency"], ["load_context", "guard", "analyze", "emergency"]],
    "default": ["load_context", "guard", "analyze", "retrieve", "answer"],
}
BOOKING_INTENTS = {"new_booking", "booking_update", "confirm_booking", "decline_booking", "abandon_booking"}
ANSWER_INTENTS = {"clinic_info", "recommendation", "booking_query", "cancel_existing", "out_of_scope"}


# ------------------------------------------------------------------------------------------------ http
def call(base: str, method: str, path: str, body: dict | None = None, timeout: int = 180) -> Any:
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.load(res)


def chat(base: str, session: str, message: str) -> dict:
    return call(base, "POST", "/chat", {"message": message, "sessionId": session, "patientId": PATIENT})


def wait_for(fn, timeout: float = 45.0, every: float = 1.5):
    end = time.time() + timeout
    while True:
        value = fn()
        if value or time.time() > end:
            return value
        time.sleep(every)


# ------------------------------------------------------------------------------------------------ trajectory
def _as_messages(user: str, steps: list[str]):
    from langchain_core.messages import AIMessage, HumanMessage

    calls = [{"name": s, "args": {}, "id": f"c{i}", "type": "tool_call"} for i, s in enumerate(steps)]
    return [HumanMessage(user), AIMessage(content="", tool_calls=calls)]


_EVALUATORS: dict[str, Any] = {}


def trajectory(user: str, r: dict) -> dict:
    """agentevals trajectory match: the node path must equal the intent's route (strict), and the tools the case
    expects must have been called (superset)."""
    from agentevals.trajectory.match import create_trajectory_match_evaluator

    if not _EVALUATORS:
        _EVALUATORS["strict"] = create_trajectory_match_evaluator(trajectory_match_mode="strict", tool_args_match_mode="ignore")
        _EVALUATORS["superset"] = create_trajectory_match_evaluator(trajectory_match_mode="superset", tool_args_match_mode="ignore")
    steps = [t["step"] for t in r["trace"]]
    intent = r["intent"]
    key = "booking" if intent in BOOKING_INTENTS else intent if intent in ROUTES else "default"
    refs = ROUTES[key] if isinstance(ROUTES[key][0], list) else [ROUTES[key]]
    ok = any(_EVALUATORS["strict"](outputs=_as_messages(user, steps), reference_outputs=_as_messages(user, ref))["score"]
             for ref in refs)
    return {"route_ok": bool(ok), "path": steps}


def tools_superset(user: str, r: dict, expected: list[str]) -> bool:
    if not expected:
        return True
    got = [t["name"] for t in r["tool_calls"]]
    res = _EVALUATORS["superset"](outputs=_as_messages(user, got), reference_outputs=_as_messages(user, expected))
    return bool(res["score"])


# ------------------------------------------------------------------------------------------------ judge
class Judge:
    """LLM-as-judge for groundedness (openevals). Uses the AI service's model settings from ai-service/.env unless
    JUDGE_MODEL / JUDGE_BASE_URL / JUDGE_API_KEY are set; temperature 0. Calibrate with evals/judge_calibration.py."""

    def __init__(self) -> None:
        from langchain_openai import ChatOpenAI
        from openevals.llm import create_llm_as_judge
        from openevals.prompts import RAG_GROUNDEDNESS_PROMPT

        class FunctionCallingChat(ChatOpenAI):
            """DeepSeek rejects response_format=json_schema (openevals' default); tool calling works everywhere."""

            def with_structured_output(self, schema, *, method="function_calling", **kw):  # type: ignore[override]
                return super().with_structured_output(schema, method=method, **kw)

        env = _ai_env()
        model = FunctionCallingChat(model=os.environ.get("JUDGE_MODEL", env.get("LLM_MODEL", "deepseek-chat")),
                           base_url=os.environ.get("JUDGE_BASE_URL", env.get("LLM_BASE_URL", "https://api.deepseek.com")),
                           api_key=os.environ.get("JUDGE_API_KEY", env.get("LLM_API_KEY")), temperature=0, max_retries=2)
        self.name = model.model_name
        self.grounded = create_llm_as_judge(prompt=RAG_GROUNDEDNESS_PROMPT, feedback_key="groundedness", judge=model)

    def groundedness(self, answer: str, context: str) -> dict:
        res = self.grounded(context=context, outputs=answer)
        return {"score": bool(res["score"]), "comment": (res.get("comment") or "")[:1500]}


def _ai_env() -> dict[str, str]:
    env: dict[str, str] = {}
    p = HERE.parent / "ai-service" / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def judge_context(base: str, r: dict, catalog: dict, profile: dict) -> str:
    """What the answer may rely on. Preferred: the exact facts the answer model saw (`grounding`, returned when the AI
    service runs with EVAL_EXPOSE_GROUNDING=true). Fallback: cited records + clinics + patient record + bookings,
    which can raise false alarms (the judge misses system facts and code-rendered dates)."""
    if r.get("grounding"):
        return "\n".join(r["grounding"])[:16000]
    by_id = {x["id"]: x for kind in ("clinics", "doctors", "treatments") for x in catalog[kind]}
    cited = [by_id[c] for c in r["citations"] if c in by_id]
    keep = {k: profile.get(k) for k in ("patientId", "firstName", "allergies", "treatmentHistory")}
    bookings = call(base, "GET", f"/bookings?patientId={PATIENT}")
    return json.dumps({"cited_records": cited, "clinics": catalog["clinics"], "patient": keep,
                       "bookings": bookings}, ensure_ascii=False)[:12000]


# ------------------------------------------------------------------------------------------------ cases
def run_scenario(base: str, case: dict, judge: Judge | None, catalog: dict, profile: dict) -> dict:
    session = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    turns, fails = [], []
    for i, exp in enumerate(case["turns"], 1):
        try:
            r = chat(base, session, exp["user"])
        except urllib.error.URLError as exc:
            fails.append(f"turn {i}: request failed: {exc}")
            break
        f = check_turn(exp, r)
        tr = trajectory(exp["user"], r)
        if not tr["route_ok"]:
            f.append(f"route {tr['path']} does not match intent {r['intent']}")
        if not tools_superset(exp["user"], r, exp.get("tools", [])):
            f.append("expected tools missing (trajectory superset)")
        turn = {"turn": i, "intent": r["intent"], "route_ok": tr["route_ok"], "latency_ms": r["latency_ms"],
                "tokens": r["usage"]["input_tokens"] + r["usage"]["output_tokens"], "llm_calls": r["usage"]["llm_calls"],
                "tokens_in": r["usage"]["input_tokens"], "tokens_out": r["usage"]["output_tokens"],
                "trace_id": r.get("trace_id"), "fails": f}
        if judge and r["intent"] in ANSWER_INTENTS and not r.get("booking_created"):
            g = judge.groundedness(r["reply"], judge_context(base, r, catalog, profile))
            turn["grounded"], turn["judge_comment"] = g["score"], g["comment"]
            turn["grounding_source"] = "exact" if r.get("grounding") else "reconstructed"
            if not g["score"]:
                f.append(f"judge: not grounded ({g['comment'][:120]})")
        fails += [f"turn {i}: {x}" for x in f]
        turns.append(turn)
    return {"id": case["id"], "suite": "scenarios", "passed": not fails, "fails": fails, "turns": turns}


def run_memory_case(base: str, case: dict) -> dict:
    """Multi-session case: each step may open a new session, end it, and wait for background memory writes."""
    sessions: dict[str, str] = {}
    turns, fails = [], []
    for i, step in enumerate(case["steps"], 1):
        sid = sessions.setdefault(step.get("session", "a"), f"mem-{case['id']}-{step.get('session', 'a')}-{uuid.uuid4().hex[:5]}")
        if step.get("end_session"):
            call(base, "POST", f"/chat/sessions/{sid}/end?patientId={PATIENT}")
            ok = wait_for(lambda: call(base, "GET", f"/patients/{PATIENT}/memory").get("episodes"))
            if not ok:
                fails.append(f"step {i}: episode not written")
            continue
        if step.get("wait_value"):  # background writes are eventually consistent: poll until they land
            want = step["wait_value"]

            def landed() -> bool:
                prof = call(base, "GET", f"/patients/{PATIENT}/memory").get("profile", {})
                return all((prof.get(k) or {}).get("value") == v for k, v in want.items())
            if not wait_for(landed):
                fails.append(f"step {i}: preference {want} not written")
            continue
        r = chat(base, sid, step["user"])
        reply = r["reply"].lower()
        f = []
        expected_any = [_today_tokens(x) for x in step.get("reply_any", [])]
        if expected_any and not any(s.lower() in reply for s in expected_any):
            f.append(f"reply has none of {expected_any}")
        for s in step.get("reply_lacks", []):
            if s.lower() in reply:
                f.append(f"reply should not contain '{s}'")
        if step.get("memory_recalled") and "memory:" not in r["trace"][0]["detail"]:
            f.append("long-term memory was not recalled")
        turns.append({"turn": i, "intent": r["intent"], "latency_ms": r["latency_ms"],
                      "tokens": r["usage"]["input_tokens"] + r["usage"]["output_tokens"],
                      "tokens_in": r["usage"]["input_tokens"], "tokens_out": r["usage"]["output_tokens"],
                      "llm_calls": r["usage"]["llm_calls"], "trace_id": r.get("trace_id"), "fails": f})
        fails += [f"step {i}: {x}" for x in f]
    return {"id": case["id"], "suite": "memory", "category": case["category"], "passed": not fails, "fails": fails,
            "turns": turns}


def _today_tokens(text: str) -> str:
    """Episode dates use the real date in the clinic's timezone (America/New_York by default)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    d = datetime.now(ZoneInfo(_ai_env().get("CLINIC_TIMEZONE", "America/New_York"))).date()
    return (text.replace("{today_iso}", d.isoformat()).replace("{today_long}", f"{d.strftime('%B')} {d.day}")
            .replace("{today_short}", f"{d.strftime('%b')} {d.day}"))


# ------------------------------------------------------------------------------------------------ metrics
def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else 0.0


def summarize(results: list[list[dict]], price_in: float | None, price_out: float | None) -> dict:
    """results[k] = all cases of repetition k."""
    k = len(results)
    ids = [c["id"] for c in results[0]]
    per_case = {cid: [next(c for c in rep if c["id"] == cid)["passed"] for rep in results] for cid in ids}
    turns = [t for rep in results for c in rep for t in c["turns"]]
    lat = [t["latency_ms"] for t in turns]
    tok = [t["tokens"] for t in turns]
    graded = [t["grounded"] for t in turns if "grounded" in t]
    routes = [t["route_ok"] for t in turns if "route_ok" in t]
    s = {
        "repetitions": k,
        "cases": len(ids),
        "pass@1": round(statistics.mean(sum(v) / k for v in per_case.values()), 3),
        f"pass^{k}": round(sum(all(v) for v in per_case.values()) / len(ids), 3),
        "route_accuracy": round(sum(routes) / len(routes), 3) if routes else None,
        "groundedness": round(sum(graded) / len(graded), 3) if graded else None,
        "turns": len(turns),
        "latency_ms_p50": int(pct(lat, 0.5)), "latency_ms_p95": int(pct(lat, 0.95)), "latency_ms_max": int(max(lat or [0])),
        "tokens_per_turn": int(statistics.mean(tok)) if tok else 0,
        "llm_calls_per_turn": round(statistics.mean(t["llm_calls"] for t in turns), 2) if turns else 0,
    }
    by_cat: dict[str, list[bool]] = {}
    for rep in results:
        for c in rep:
            if "category" in c:
                by_cat.setdefault(c["category"], []).append(c["passed"])
    if by_cat:
        s["memory_by_category"] = {cat: f"{sum(v)}/{len(v)}" for cat, v in sorted(by_cat.items())}
    if price_in is not None and price_out is not None and turns:
        cost = [(t["tokens_in"] * price_in + t["tokens_out"] * price_out) / 1e6 for t in turns]
        s["cost_per_turn_usd"] = round(statistics.mean(cost), 6)  # list price, ignores provider cache discounts
    s["flaky_cases"] = [cid for cid, v in per_case.items() if any(v) and not all(v)]
    s["failing_cases"] = [cid for cid, v in per_case.items() if not any(v)]
    return s


def gate(summary: dict, baseline: dict | None, gates: dict, case_ids: list[str]) -> tuple[list[str], list[str]]:
    """Absolute thresholds, plus 'no worse than baseline' within a tolerance (LLM runs are noisy).
    Returns (failures, not_measured): a gated metric this run did not measure (e.g. groundedness without --judge)
    is reported, never silently treated as passing."""
    out, missing = [], []
    for key, rule in gates.get("min", {}).items():
        if summary.get(key) is None:
            missing.append(key)
        elif summary[key] < rule:
            out.append(f"{key} {summary[key]} < {rule}")
    for key, rule in gates.get("max", {}).items():
        if summary.get(key) is None:
            missing.append(key)
        elif summary[key] > rule:
            out.append(f"{key} {summary[key]} > {rule}")
    if baseline:
        b = baseline["summary"]
        base_ids = sorted(c["id"] for c in baseline["results"][0])
        if base_ids != sorted(case_ids) or b.get("repetitions") != summary.get("repetitions"):
            out.append("baseline is not comparable (different cases or repetitions): re-run with the same options")
            return out, missing
        for key, tol in gates.get("no_worse_than_baseline", {}).items():
            if key in b and key in summary and b[key] is not None and summary[key] is not None:
                worse = summary[key] > b[key] * (1 + tol) if key.startswith(("latency", "tokens", "llm_calls", "cost")) \
                    else summary[key] < b[key] - tol
                if worse:
                    out.append(f"{key} regressed: {summary[key]} vs baseline {b[key]} (tolerance {tol})")
    return out, missing


# ------------------------------------------------------------------------------------------------ langfuse
def to_langfuse(run_name: str, results: list[list[dict]], summary: dict) -> str | None:
    """Records the run as a Langfuse dataset run: one item per case, scores per item, run-level scores."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    from langfuse import Evaluation, get_client

    lf = get_client()
    name = "clinicpilot-agent-eval"
    try:
        lf.get_dataset(name)
    except Exception:  # noqa: BLE001 - first run creates it
        lf.create_dataset(name=name, description="ClinicPilot scenario + memory cases (evals/cases.json, memory_cases.json)")
    first = {c["id"]: c for c in results[0]}
    for cid, c in first.items():
        lf.create_dataset_item(dataset_name=name, id=f"{name}-{cid}", input={"case": cid, "suite": c["suite"]},
                               metadata={"category": c.get("category")})
    reps = {cid: [next(x for x in rep if x["id"] == cid) for rep in results] for cid in first}

    def task(*, item, **_):  # the cases already ran against the live stack; the experiment records their outcome
        cid = item.input["case"]
        return {"runs": [{"passed": r["passed"], "fails": r["fails"][:5],
                          "trace_ids": [t.get("trace_id") for t in r["turns"] if t.get("trace_id")]} for r in reps[cid]]}

    def passed_all(*, output, **_):
        return Evaluation(name="pass_all_k", value=float(all(r["passed"] for r in output["runs"])))

    def pass_rate(*, output, **_):
        runs = output["runs"]
        return Evaluation(name="pass_rate", value=sum(r["passed"] for r in runs) / len(runs),
                          comment="; ".join(f for r in runs for f in r["fails"])[:500] or None)

    def run_level(**_):
        return [Evaluation(name=k, value=float(v)) for k, v in summary.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)]

    # only this run's cases: the dataset also holds items from earlier runs with other --only / --suite options
    items = [i for i in lf.get_dataset(name).items if i.input.get("case") in reps]
    res = lf.run_experiment(name="agent-eval", run_name=run_name, data=items, task=task,
                            evaluators=[passed_all, pass_rate], run_evaluators=[run_level], max_concurrency=4,
                            metadata={k: v for k, v in summary.items() if not isinstance(v, (list, dict))})
    lf.flush()
    return getattr(res, "dataset_run_url", None)


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:3000/api")
    ap.add_argument("--suite", choices=["scenarios", "memory", "all"], default="all")
    ap.add_argument("--repeat", type=int, default=1, help="k runs per case: reports pass@1 and pass^k")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge groundedness on answer turns")
    ap.add_argument("--only", default=None, help="comma-separated case ids")
    ap.add_argument("--baseline", default=None, help="results json of a previous run to compare against")
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (for cost per turn)")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--save-baseline", action="store_true", help="also write evals/results/baseline.json")
    ap.add_argument("--no-reset", action="store_true", help="do not reset demo bookings before each case")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    scenarios = json.loads((HERE / "cases.json").read_text()) if args.suite in ("scenarios", "all") else []
    memory = json.loads((HERE / "memory_cases.json").read_text()) if args.suite in ("memory", "all") else []
    if args.only:
        keep = set(args.only.split(","))
        scenarios = [c for c in scenarios if c["id"] in keep]
        memory = [c for c in memory if c["id"] in keep]
    judge = Judge() if args.judge else None
    catalog = call(base, "GET", "/catalog")
    profile = call(base, "GET", f"/patients/{PATIENT}")

    def isolate() -> None:
        if not args.no_reset:  # bookings from earlier cases would (correctly) be rejected as duplicates
            subprocess.run(["npm", "run", "db:demo-reset"], cwd=HERE.parent / "backend", check=True, capture_output=True)
        call(base, "DELETE", f"/patients/{PATIENT}/memory")

    results: list[list[dict]] = []
    started = time.time()
    for k in range(args.repeat):
        rep = []
        for case in scenarios:
            isolate()
            res = run_scenario(base, case, judge, catalog, profile)
            rep.append(res)
            print(f"[{k + 1}/{args.repeat}] {'PASS' if res['passed'] else 'FAIL'}  {case['id']}"
                  + ("" if res["passed"] else "\n      " + "\n      ".join(res["fails"])), flush=True)
        for case in memory:
            isolate()
            res = run_memory_case(base, case)
            rep.append(res)
            print(f"[{k + 1}/{args.repeat}] {'PASS' if res['passed'] else 'FAIL'}  memory/{case['category']}/{case['id']}"
                  + ("" if res["passed"] else "\n      " + "\n      ".join(res["fails"])), flush=True)
        results.append(rep)
    call(base, "DELETE", f"/patients/{PATIENT}/memory")

    summary = summarize(results, args.price_in, args.price_out)
    summary["judge_model"] = judge.name if judge else None
    summary["judge_context"] = ("exact answer facts" if any(t.get("grounding_source") == "exact" for rep in results
                                                            for c in rep for t in c["turns"]) else "reconstructed")
    summary["duration_s"] = int(time.time() - started)
    gates = json.loads((HERE / "gates.json").read_text())
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    problems, not_measured = gate(summary, baseline, gates, [c["id"] for c in results[0]])
    summary["gates_not_measured"] = not_measured

    run_name = f"agent-eval-{time.strftime('%Y%m%d-%H%M%S')}"
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    payload = {"run": run_name, "base": base, "summary": summary, "gate_failures": problems, "results": results}
    (out / f"{run_name}.json").write_text(json.dumps(payload, indent=1))
    if args.save_baseline:
        (out / "baseline.json").write_text(json.dumps(payload, indent=1))
    url = to_langfuse(run_name, results, summary)
    write_report(out / "agent_eval_latest.md", run_name, base, summary, problems, results, baseline, url)

    print("\n" + json.dumps(summary, indent=2))
    print("\nGATES: " + ("PASS" if not problems else "FAIL\n  " + "\n  ".join(problems)))
    if not_measured:
        print(f"Not measured in this run (gates not applied): {', '.join(not_measured)}")
    if url:
        print(f"Langfuse dataset run: {url}")
    return 1 if problems else 0


def write_report(path: Path, run: str, base: str, s: dict, problems: list[str], results: list[list[dict]],
                 baseline: dict | None, url: str | None) -> None:
    rows = []
    for c in results[0]:
        runs = [next(x for x in rep if x["id"] == c["id"]) for rep in results]
        ok = sum(r["passed"] for r in runs)
        why = "; ".join(sorted({f for r in runs for f in r["fails"]}))[:300]
        rows.append(f"| {c.get('category') or c['suite']} | {c['id']} | {ok}/{len(runs)} | {why} |")
    b = baseline["summary"] if baseline else {}
    lines = [
        "# ClinicPilot AI - agent evaluation", "",
        f"Run `{run}` against `{base}`" + (f" · [Langfuse dataset run]({url})" if url else ""), "",
        "| Metric | Value | Baseline |", "|---|---|---|",
        *[f"| {k} | {v} | {b.get(k, '')} |" for k, v in s.items() if not isinstance(v, (list, dict))],
        "", f"Memory by category: {s.get('memory_by_category', {})}",
        f"Flaky cases: {s['flaky_cases'] or 'none'} · failing cases: {s['failing_cases'] or 'none'}", "",
        "**Gates: " + ("PASS**" if not problems else "FAIL**\n\n- " + "\n- ".join(problems)), "",
        "| Suite | Case | Passed | Issues |", "|---|---|---|---|", *rows, "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
