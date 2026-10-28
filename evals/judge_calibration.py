"""Calibrates the groundedness judge used by evals/agent_eval.py before trusting its scores.

Builds a labelled set from the clinic catalog: faithful answers (labelled grounded) and the same answers with one
fact changed - price, opening hours, specialty, a service that does not exist (labelled not grounded). Labels are
known by construction, so the judge's accuracy, recall on errors and Cohen's kappa can be measured. Human-labelled
real answers can be added to evals/judge_labels.json ({"context": ..., "answer": ..., "grounded": true|false}).

    ai-service/.venv/bin/python evals/judge_calibration.py --base http://localhost:3000/api
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from agent_eval import Judge, call

HERE = Path(__file__).resolve().parent


def build_items(catalog: dict, seed: int = 7) -> list[dict]:
    rnd = random.Random(seed)
    items: list[dict] = []
    clinics = {c["id"]: c for c in catalog["clinics"]}
    for d in catalog["doctors"]:
        c = clinics[d["clinicId"]]
        ctx = json.dumps({"doctor": d, "clinic": c})
        fee = d["consultationFeeUsd"]
        good = f"{d['fullName']} is a {d['specialty']} doctor at {c['name']}; the consultation fee is ${fee:.0f}."
        items.append({"context": ctx, "answer": good, "grounded": True, "kind": "doctor fact"})
        wrong_fee = fee + rnd.choice([30, 50, 70])
        items.append({"context": ctx, "answer": good.replace(f"${fee:.0f}", f"${wrong_fee:.0f}"), "grounded": False,
                      "kind": "wrong price"})
        other = rnd.choice([s for s in ("Cardiology", "Neurology", "Ophthalmology") if s != d["specialty"]])
        items.append({"context": ctx, "answer": good.replace(d["specialty"], other), "grounded": False,
                      "kind": "wrong specialty"})
    for c in catalog["clinics"]:
        ctx = json.dumps({"clinic": c})
        good = f"{c['name']} is open {c['operatingHours']}. Phone: {c['phone']}."
        items.append({"context": ctx, "answer": good, "grounded": True, "kind": "clinic hours"})
        items.append({"context": ctx, "answer": good.replace(c["operatingHours"], "Mon-Sun: 7:00 AM - 11:00 PM"),
                      "grounded": False, "kind": "wrong hours"})
        items.append({"context": ctx, "answer": good + " We also offer free MRI scans on Saturdays.", "grounded": False,
                      "kind": "invented service"})
    for t in catalog["treatments"]:
        ctx = json.dumps({"treatment": t})
        good = f"{t['name']} takes {t['durationMinutes']} minutes and costs ${t['priceUsd']:.0f}."
        items.append({"context": ctx, "answer": good, "grounded": True, "kind": "treatment fact"})
        items.append({"context": ctx, "answer": good.replace(f"{t['durationMinutes']} minutes", f"{t['durationMinutes'] * 3} minutes"),
                      "grounded": False, "kind": "wrong duration"})
    extra = HERE / "judge_labels.json"
    if extra.exists():
        items += [dict(x, kind="human-labelled") for x in json.loads(extra.read_text())]
    return items


LIVE_QUESTIONS = [
    "What are the opening hours of Metro Care?",
    "How much is a flu shot?",
    "Which doctors speak Spanish?",
    "What is the cancellation policy at Sunset Wellness?",
    "I've had an itchy rash on my arm for a week.",
    "Where is Sunset Wellness located?",
]


def live_items(base: str) -> list[dict]:
    """Realistic context: the exact rules + facts + conversation the agent used (AI service in evaluation mode,
    EVAL_EXPOSE_GROUNDING=true). Each real reply is a positive; the same reply with one number changed must be
    flagged. Checks that the richer context does not make the judge lenient."""
    import re
    import uuid

    from agent_eval import chat

    out = []
    for q in LIVE_QUESTIONS:
        r = chat(base, f"calib-{uuid.uuid4().hex[:6]}", q)
        if not r.get("grounding"):
            raise SystemExit("AI service is not in evaluation mode (EVAL_EXPOSE_GROUNDING=true)")
        ctx = "\n".join(r["grounding"])
        out.append({"context": ctx, "answer": r["reply"], "grounded": True, "kind": "live reply"})
        m = re.search(r"\$(\d+(?:\.\d+)?)|\b(\d{1,2}):(\d{2})", r["reply"])
        if m:
            wrong = (r["reply"][:m.start()] + (f"${float(m.group(1)) + 35:.0f}" if m.group(1) else f"{(int(m.group(2)) + 3) % 12 or 12}:{m.group(3)}")
                     + r["reply"][m.end():])
            out.append({"context": ctx, "answer": wrong, "grounded": False, "kind": "live reply, one number changed"})
        else:
            out.append({"context": ctx, "answer": r["reply"] + " We also offer free MRI scans every Saturday.",
                        "grounded": False, "kind": "live reply + invented service"})
    return out


def kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:3000/api")
    ap.add_argument("--live", action="store_true", help="add realistic items from the running agent (evaluation mode)")
    args = ap.parse_args()
    items = build_items(call(args.base.rstrip("/"), "GET", "/catalog"))
    if args.live:
        items += live_items(args.base.rstrip("/"))
    judge = Judge()
    pairs, wrong = [], []
    for it in items:
        got = judge.groundedness(it["answer"], it["context"])["score"]
        pairs.append((it["grounded"], got))
        if got != it["grounded"]:
            wrong.append(f"{it['kind']}: expected {it['grounded']}, judge said {got}: {it['answer'][:90]}")
    n = len(pairs)
    neg = [(a, b) for a, b in pairs if not a]
    by_kind: dict[str, list[bool]] = {}
    for it, (a, b) in zip(items, pairs):
        by_kind.setdefault(it["kind"], []).append(a == b)
    report = {
        "judge_model": judge.name, "items": n,
        "by_kind": {k: f"{sum(v)}/{len(v)}" for k, v in by_kind.items()},
        "accuracy": round(sum(a == b for a, b in pairs) / n, 3),
        "error_recall": round(sum(not b for _, b in neg) / len(neg), 3) if neg else None,  # wrong facts caught
        "false_alarm_rate": round(sum(not b for a, b in pairs if a) / max(1, sum(a for a, _ in pairs)), 3),
        "cohens_kappa": round(kappa(pairs), 3),
        "disagreements": wrong,
    }
    out = HERE / "results" / "judge_calibration.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
