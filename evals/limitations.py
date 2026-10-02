"""The known limitations, measured on purpose with fixed cases instead of waiting for them to show up by chance.

  L1  reworded duplicate twins aren't merged       live: replays run 335ba89's final Editor input (#4/#24)
  L2  posting attribution near its cutoff           offline: the rule vs 137 hand-labeled claims
  L3  "verified" can mean source-matches-source      live: independent judge on 10 verified claims with ground truth

Usage: python evals/limitations.py [l1] [l2] [l3]    (default: all). Writes evals/results/limitations_<timestamp>.json
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import judge
import graph as g

EVALS = Path(__file__).resolve().parent
L1_REPLAYS = 1  # 3 planned; cut to 1 for budget (2026-09-25)


def load_jsonl(name):
    return [json.loads(line) for line in (EVALS / "labels" / name).read_text(encoding="utf-8").splitlines()]


def l1_duplicate_twins():
    fixture = json.loads((EVALS / "fixtures" / "l1_twins_editor_input.json").read_text(encoding="utf-8"))
    a, b = fixture["twin_a"], fixture["twin_b"]
    replays = []
    for _ in range(L1_REPLAYS):
        out = g.editor(fixture["editor_input"])
        claims = {c["claim_index"]: c for cs in out["verified_findings"].values() for c in cs}
        checked = {k["claim_index"] for k in out["trace"][-1]["checks"]}
        replays.append({
            "both_twins_shown": a in claims and b in claims,
            "twin_a_status": claims.get(a, {}).get("status"), "twin_b_status": claims.get(b, {}).get("status"),
            "twin_a_checked": a in checked, "twin_b_checked": b in checked,
            # The over-claim the strict rule closed: a twin verified without a check of its own.
            "verified_without_own_check": sum(c["status"] == "verified" and i not in checked for i, c in claims.items()),
            # A NIM-answered replay isn't evidence about Claude.
            "providers": {**out["trace"][-1]["providers"],
                          "checks": sorted({k["provider"] for k in out["trace"][-1]["checks"]})},
        })
    return {"source_run": fixture["source_run"], "twins": [a, b], "pre_fix_baseline": "#4 verified via #24's check",
            "replays": replays}


def l2_posting_attribution():
    labels = load_jsonl("l2_posting_attribution.jsonl")

    def confusion(pred_key, items):
        tp = sum(i["gold_from_posting"] and i[pred_key] for i in items)
        fp = sum(not i["gold_from_posting"] and i[pred_key] for i in items)
        fn = sum(i["gold_from_posting"] and not i[pred_key] for i in items)
        return {"n": len(items), "tp": tp, "fp": fp, "fn": fn,
                "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                "recall": round(tp / (tp + fn), 3) if tp + fn else None}

    for i in labels:  # stage 1 of the rule: wording restates the posting (needs only the claim text)
        i["stage1"] = i["overlap"] >= g.POSTING_MATCH
    live = [i for i in labels if i["live_rule_flagged"] is not None]  # runs made after the rule existed
    errors = [{"id": i["id"], "overlap": i["overlap"], "gold": i["gold_from_posting"], "claim": i["claim"][:140],
               "why": i["reasoning"][:200]} for i in labels if i["stage1"] != i["gold_from_posting"]]
    return {"cutoff": g.POSTING_MATCH, "stage1_all_runs": confusion("stage1", labels),
            "live_rule_post_fix_runs": confusion("live_rule_flagged", live),
            "errors": sorted(errors, key=lambda e: -e["overlap"]),
            "flagged_labels": [i["id"] for i in labels if i["flagged"]]}


def l3_source_vs_fact():
    rows = []
    for item in load_jsonl("l3_verified_ground_truth.jsonl"):
        verdict, _ = judge.recheck(item["claim"], "Sony Interactive Entertainment")
        rows.append({"id": item["id"], "claim": item["claim"][:120], "true_to_source": item["true_to_source"],
                     "true_as_fact": item["true_as_fact"], "independent_judge": verdict.verdict,
                     "judge_reasoning": verdict.reasoning, "flagged": item["flagged"]})
    return {"n": len(rows),
            "true_to_source": sum(r["true_to_source"] for r in rows),
            "true_as_fact": sum(r["true_as_fact"] is True for r in rows),
            "not_true_as_fact": sum(r["true_as_fact"] is False for r in rows),
            "unestablished": sum(r["true_as_fact"] is None for r in rows),
            # Kept apart on purpose: "provably wrong" and "couldn't establish either way" are different findings.
            "judge_supported_but_false": [r["id"] for r in rows
                                          if r["independent_judge"] == "supported" and r["true_as_fact"] is False],
            "judge_supported_but_unestablished": [r["id"] for r in rows
                                                  if r["independent_judge"] == "supported" and r["true_as_fact"] is None],
            "rows": rows}


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["l1", "l2", "l3"]
    if {"l1", "l3"} & set(wanted):
        judge.meter_pipeline()
    results = {k: {"l1": l1_duplicate_twins, "l2": l2_posting_attribution, "l3": l3_source_vs_fact}[k]() for k in wanted}
    results["cost"] = {"pipeline": round(judge.cost(g.MODEL), 3), "judge": round(judge.cost(judge.JUDGE_MODEL), 3)}
    out = EVALS / "results" / f"limitations_{datetime.now().strftime('%Y-%m-%d_%H%M')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "l3"} | (
        {"l3": {k: v for k, v in results["l3"].items() if k != "rows"}} if "l3" in results else {}),
        ensure_ascii=False, indent=1)[:4000])
    print(f"\nWrote {out}")
