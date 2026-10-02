"""Smallest live check of the contradicted / from_posting fixes: the Editor's review step alone, on one claim each.

  CaseGuard #2  "33 categories", its saved check found "30+"   -> expect unverified, not contradicted
  Sony #10      posting text plus 3 product names it never says -> expect unverified, not from_posting

The questions and answers are replayed from eval run 2026-09-25_1341 (no search, no answering calls), so each replay is
one Sonnet call. Usage: EVAL_BUDGET_USD=0.40 python evals/fix_check.py [replays per claim, default 1]
"""
import json
import sys
from pathlib import Path

import judge
import graph as g

EVALS = Path(__file__).resolve().parent
STATES = EVALS / "results" / "2026-09-25_1341" / "states"
CASES = [("caseguard", "CaseGuard", 2, "contradicted"), ("sony_play_station", "Sony Play Station", 10, "from_posting")]


def replay(key, company, index):
    state = json.loads((STATES / f"{key}.json").read_text(encoding="utf-8"))
    finding = state["raw_findings"][index]
    checks = [k for k in [t for t in state["trace"] if t["node"] == "editor"][-1]["checks"] if k["claim_index"] == index]
    plan = g.VerificationPlan(questions=[g.VerificationQuestion(claim_index=0, question=k["question"]) for k in checks])
    answers = {k["question"]: g.Answer(answer=k["answer"], source_url=k["source_url"]) for k in checks}
    results = {k["question"]: k["results"] for k in checks}
    live = g.ask

    def ask(model, prompt, attempts=1):  # only the review is a live call
        if model is g.VerificationPlan:
            return plan, "replayed"
        if model is g.Answer:
            return answers[prompt.split("Question: ")[1].split("\n")[0]], "replayed"
        return live(model, prompt, attempts)

    g.ask, g.search = ask, (lambda q: results[q])
    try:
        out = g.editor({**state, "company_name": company, "raw_findings": [finding], "iteration_count": 3})
    finally:
        g.ask = live
    return next(c for cs in out["verified_findings"].values() for c in cs), out["trace"][-1]["providers"]["review"]


if __name__ == "__main__":
    replays = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    judge.meter_pipeline()
    for key, company, index, old_status in CASES:
        for n in range(replays):
            c, provider = replay(key, company, index)
            print(f"{company} #{index} (was {old_status}) replay {n + 1}: {c['status']}  [{provider}]\n"
                  f"  unconfirmed: {c.get('unconfirmed_specifics')}  conflicting: {c.get('conflicting_values')}\n"
                  f"  note: {c['note']}", flush=True)
    print(f"\nSpend: ${judge.cost():.3f}")
