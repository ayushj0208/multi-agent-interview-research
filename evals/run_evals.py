"""Phase 4 evals: live runs on a fixed test set, scored by an independent Opus judge, Ragas and DeepEval.

Usage: python evals/run_evals.py [company_key ...] [--reuse <states dir>]    (default: all six)
Set EVAL_BUDGET_USD to stop the run once metered spend reaches it.
Writes evals/results/<timestamp>/claims.jsonl, runs.jsonl, states/<company>.json and summary.md.
"""
import asyncio
import json
import re
import sys
import time
import types
from collections import Counter
from datetime import datetime
from pathlib import Path

import judge  # sets up sys.path and DeepEval telemetry opt-out before anything else imports them
import graph as g

# Ragas 0.4.3 imports a Vertex AI module that langchain-community 0.4 removed; only the non-LLM ToolCallF1 is used
# here, so a stand-in module is enough. Remove once Ragas stops importing it.
for _name in ("langchain_community.chat_models.vertexai", "langchain_community.llms.vertexai"):
    _stub = types.ModuleType(_name)
    _stub.ChatVertexAI = _stub.VertexAI = type("Unavailable", (), {})
    sys.modules.setdefault(_name, _stub)

from deepeval.metrics import FaithfulnessMetric, HallucinationMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from ragas.dataset_schema import MultiTurnSample  # noqa: E402
from ragas.messages import AIMessage, HumanMessage, ToolCall  # noqa: E402
from ragas.metrics import ToolCallF1  # noqa: E402

EVALS = Path(__file__).resolve().parent
COMPANIES = {"sony_play_station": "Sony Play Station", "scout_ai": "Scout AI", "caseguard": "CaseGuard",
             "stripe": "Stripe", "databricks": "Databricks", "notion": "Notion"}
NEVER_CHECKED_SAMPLE = 5  # unverified claims the 8-question budget skipped, re-checked per run to size budget misses
ATTRIBUTION = re.compile(r"\b(according to|per|reports?|reported|says|lists?|shows?)\b", re.I)


# --- Per-claim facts from a finished pipeline run (no API calls) -------------------------------------------------

def final_checks(state):
    return [t for t in state["trace"] if t["node"] == "editor"][-1]["checks"]


def unverified_bucket(claim, checked):
    """Separates budget misses (never checked) from checks that ran but couldn't confirm everything."""
    if claim["claim_index"] not in checked:
        return "never_checked"
    return "partial" if claim.get("unconfirmed_specifics") else "inconclusive"


def source_type(claim, company):
    if not claim["source_url"]:
        return "posting" if claim["status"] == "from_posting" else "none"
    return {2: "company_or_reputable", 0: "other", -1: "aggregator", -2: "low_quality"}[
        g.source_quality(claim["source_url"], company)]


def attribution_scoped(claim_text):
    """'X says/lists/reports ...' claims can be verified against X while still being wrong as fact."""
    host_names = [h.split(".")[0] for h in g.AGGREGATOR_HOSTS]
    return bool(ATTRIBUTION.search(claim_text)) or any(h in claim_text.lower() for h in host_names)


UNRECORDED = "unrecorded"  # traces from before providers were recorded


def claim_providers(state):
    """Which providers produced each claim: its extraction, the final Editor's question/review calls, its own check.

    A claim any NIM answer touched is kept out of the Claude-pipeline scores instead of silently mixing in."""
    extracted_by = []
    for t in state["trace"]:
        if t["node"] == "researcher":
            extracted_by += [t.get("providers", {}).get("extract") or UNRECORDED] * t["new_findings"]
    editor = [t for t in state["trace"] if t["node"] == "editor"][-1]
    reviewed_by = set(editor.get("providers", {"review": UNRECORDED}).values())

    def providers(i):
        own = {k.get("provider", UNRECORDED) for k in editor["checks"] if k["claim_index"] == i}
        return sorted({extracted_by[i] if 0 <= i < len(extracted_by) else UNRECORDED} | reviewed_by | own)
    return providers


def run_providers(trace):
    used = set()
    for t in trace:
        used |= {p for p in t.get("providers", {}).values() if p} | {k.get("provider") for k in t.get("checks", [])}
        used.add(t.get("provider"))
    used.discard(None)
    return sorted(used) or [UNRECORDED]


def claim_rows(state, company, run_key):
    checks = final_checks(state)
    checked = {k["claim_index"] for k in checks}
    posting_words = g.words(state["job_posting"])
    providers = claim_providers(state)
    rows = []
    for category, claims in state["verified_findings"].items():
        for c in claims:
            by = providers(c["claim_index"])
            rows.append({
                "providers": by, "nim_involved": "nim" in by,
                "run": run_key, "company": company, "claim_index": c["claim_index"], "claim": c["claim"],
                "category": category, "status": c["status"], "source_url": c["source_url"],
                "source_type": source_type(c, company), "own_check": c["claim_index"] in checked,
                "unconfirmed_specifics": c.get("unconfirmed_specifics", []),
                "bucket": unverified_bucket(c, checked) if c["status"] == "unverified" else None,
                "attribution_scoped": attribution_scoped(c["claim"]),
                "citation_corrected": "Citation corrected" in c["note"],
                "citation_dropped": "doesn't support this claim" in c["note"],
                "posting_match": round(g.overlap(c["claim"], posting_words), 3),
                "note": c["note"],
            })
    return sorted(rows, key=lambda r: r["claim_index"])


def select_for_recheck(rows):
    rows = [r for r in rows if not r["nim_involved"]]  # not judged: excluded from the scores anyway
    never = [r for r in rows if r["bucket"] == "never_checked"][:NEVER_CHECKED_SAMPLE]
    return [r for r in rows if r["status"] in ("verified", "contradicted") or r["bucket"] in ("partial", "inconclusive")
            ] + never


# --- Judge-scored metrics (API calls) -----------------------------------------------------------------------------

class EntityVerdict(BaseModel):
    about_company: bool = Field(description="true if the fact is about the named company itself")
    reasoning: str


def faithfulness(claim, contexts):
    if not contexts:
        return None
    metric = FaithfulnessMetric(model=judge.DeepEvalJudge(), async_mode=False, include_reason=False)
    metric.measure(LLMTestCase(input="Is this claim supported by the context?", actual_output=claim,
                               retrieval_context=contexts))
    return metric.score


def score_claims(rows, state, company):
    checks = final_checks(state)
    for r in select_for_recheck(rows):
        verdict, evidence = judge.recheck(r["claim"], company)
        r.update(recheck_verdict=verdict.verdict, recheck_reasoning=verdict.reasoning,
                 recheck_urls=verdict.evidence_urls)
        if r["status"] == "verified":  # precision of the label: own evidence vs independent evidence
            own = [x["content"] for k in checks if k["claim_index"] == r["claim_index"] for x in k.get("results", [])]
            r["faithfulness_own"] = faithfulness(r["claim"], own)
            r["faithfulness_independent"] = faithfulness(r["claim"], [x["content"] for x in evidence])
    for r in rows:
        if r["nim_involved"]:
            continue
        if r["status"] == "from_posting":
            r["posting_verdict"] = judge.posting_check(r["claim"], state["job_posting"]).verdict
        elif r["status"] == "off_target":
            v = judge.ask_judge(EntityVerdict, f"Is this fact about {company} itself, or about a different company, "
                                               f"parent, sibling, subsidiary or product line?\n\nClaim: {r['claim']}\n"
                                               f"Cited source: {r['source_url']}")
            r["judge_about_company"] = v.about_company
    return rows


class FactViolations(BaseModel):
    bullets_stating_unverified_as_fact: list[str] = Field(
        description="report sentences that present an unverified or contradicted claim as established fact")


def score_report(state):
    report = state["final_report"]
    reviewed = [f"[{c['status']}] {c['claim']}" for cs in state["verified_findings"].values() for c in cs
                if c["status"] != "off_target"]
    context = reviewed + [f"[job posting] {state['job_posting']}"]
    grounded = FaithfulnessMetric(model=judge.DeepEvalJudge(), async_mode=False, include_reason=False)
    grounded.measure(LLMTestCase(input="Write the briefing", actual_output=report, retrieval_context=context))
    # DeepEval 4.x flipped this metric: 1 = no context contradicted (a pass), not a violation rate.
    hallucination = HallucinationMetric(model=judge.DeepEvalJudge(), async_mode=False, include_reason=False)
    hallucination.measure(LLMTestCase(input="Write the briefing", actual_output=report, context=reviewed))
    tentative = [f"[{c['status']}] {c['claim']}" for cs in state["verified_findings"].values() for c in cs
                 if c["status"] in ("unverified", "contradicted")]
    violations = judge.ask_judge(FactViolations, f"""Unverified and contradicted claims the briefing was given:
{tentative}

Briefing:
{report}

List every sentence in the briefing that presents one of these unverified or contradicted claims as established fact,
with no hedging (no "reportedly", "unconfirmed", "one source says", etc.). An empty list means none.""")
    return {"report_faithfulness": grounded.score, "report_hallucination": hallucination.score,
            "unverified_stated_as_fact": len(violations.bullets_stating_unverified_as_fact),
            "unverified_stated_as_fact_examples": violations.bullets_stating_unverified_as_fact[:3]}


# --- Researcher tool calls (Ragas, no LLM) ------------------------------------------------------------------------

def planner_samples(trace):
    """One sample per Researcher pass: the categories code flagged as gaps vs the categories the planner searched."""
    gaps, samples = list(g.CATEGORIES), []
    for t in trace:
        if t["node"] == "researcher" and t.get("providers", {}).get("plan") == "nim":
            gaps = t["thin_categories"]  # a NIM-planned pass isn't the Claude planner's work; skip it
        elif t["node"] == "researcher":
            samples.append((gaps, [q["category"] for q in t["queries"]], [q["query"] for q in t["queries"]]))
            gaps = t["thin_categories"]
        elif (t["node"] == "editor" and t["decision"] == "send_back") or \
                (t["node"] == "review" and t["action"] == "send_back"):
            gaps = list(g.CATEGORIES)  # feedback reopens every category, as in researcher()
    return samples


def tool_call_f1(gaps, searched):
    if not searched:
        return 0.0 if gaps else 1.0
    sample = MultiTurnSample(
        user_input=[HumanMessage(content=f"Categories to cover: {gaps}"),
                    AIMessage(content="", tool_calls=[ToolCall(name="web_search", args={"category": c}) for c in searched])],
        reference_tool_calls=[ToolCall(name="web_search", args={"category": c}) for c in gaps])
    return float(asyncio.run(ToolCallF1().multi_turn_ascore(sample)))


def score_planner(trace, company):
    samples = planner_samples(trace)
    if not samples:
        return {"planner_tool_call_f1": None, "queries_naming_company": None}
    tokens = [t for t in re.findall(r"[a-z0-9]{3,}", company.lower())]
    queries = [q for _, _, qs in samples for q in qs]
    return {"planner_tool_call_f1": round(sum(tool_call_f1(gp, s) for gp, s, _ in samples) / len(samples), 3),
            "queries_naming_company": round(sum(any(t in q.lower() for t in tokens) for q in queries) / len(queries), 3)
            if queries else None}


# --- Run-level aggregation (no API calls) -------------------------------------------------------------------------

def ratio(rows, key, want):
    scored = [r for r in rows if key in r]
    return {"k": sum(r[key] == want for r in scored), "n": len(scored)}


def run_metrics(rows, state):
    """Scores for the Claude pipeline only: claims a NIM fallback touched are counted, then left out."""
    nim_claims = sum(r["nim_involved"] for r in rows)
    rows = [r for r in rows if not r["nim_involved"]]
    verified = [r for r in rows if r["status"] == "verified"]
    return {
        "providers_used": run_providers(state["trace"]),
        "claims_excluded_nim": nim_claims,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "unverified_buckets": dict(Counter(r["bucket"] for r in rows if r["bucket"])),
        "verified_precision": ratio(verified, "recheck_verdict", "supported"),
        "verified_attribution_scoped": sum(r["attribution_scoped"] for r in verified),
        "budget_miss": ratio([r for r in rows if r["bucket"] == "never_checked"], "recheck_verdict", "supported"),
        "truly_unconfirmable": ratio([r for r in rows if r["bucket"] in ("partial", "inconclusive")],
                                     "recheck_verdict", "unconfirmable"),
        "contradicted_agreement": ratio([r for r in rows if r["status"] == "contradicted"],
                                        "recheck_verdict", "contradicted"),
        "off_target_agreement": ratio([r for r in rows if r["status"] == "off_target"], "judge_about_company", False),
        "from_posting_precision": ratio([r for r in rows if r["status"] == "from_posting"], "posting_verdict",
                                        "supported"),
        "citations_corrected": sum(r["citation_corrected"] for r in rows),
        "citations_dropped": sum(r["citation_dropped"] for r in rows),
        "editor_send_backs": sum(t["node"] == "editor" and t["decision"] == "send_back" for t in state["trace"]),
        "researcher_passes": sum(t["node"] == "researcher" for t in state["trace"]),
    }


def fmt(r):
    return f"{r['k']}/{r['n']}" if r["n"] else "n/a"


def num(x, spec=".2f"):
    return "n/a" if x is None else format(x, spec)


def summary_table(runs):
    head = ("| Company | Claims (verified / unverified / contradicted / from posting / off-target) | Verified precision "
            "| Budget misses | Truly unconfirmable | Contradicted agreement | From-posting precision | Report faithfulness "
            "| Report consistency (1 = no contradictions) | Unverified stated as fact | Planner F1 | Providers (NIM claims excluded) "
            "| Time | Cost |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for run in runs:
        m, s, rep = run["metrics"], run["metrics"]["status_counts"], run["report"]
        mix = " / ".join(str(s.get(k, 0)) for k in ("verified", "unverified", "contradicted", "from_posting", "off_target"))
        time_s = f"{run['timing']['total_s']:.0f}s" if run["timing"] else "reused"
        cost = f"${(run['cost']['pipeline'] or 0) + run['cost']['judge']:.2f}" + ("" if run["timing"] else " (judge only)")
        lines.append(f"| {run['company']} | {mix} | {fmt(m['verified_precision'])} | {fmt(m['budget_miss'])} "
                     f"| {fmt(m['truly_unconfirmable'])} | {fmt(m['contradicted_agreement'])} "
                     f"| {fmt(m['from_posting_precision'])} | {num(rep['report_faithfulness'])} "
                     f"| {num(rep['report_hallucination'])} | {num(rep['unverified_stated_as_fact'], 'd')} "
                     f"| {num(run['planner']['planner_tool_call_f1'])} "
                     f"| {'+'.join(m['providers_used'])} ({m['claims_excluded_nim']}) | {time_s} | {cost} |")
    return "\n".join(lines)


def summary_md(name, runs, keys, reused):
    skipped = [c for k, c in COMPANIES.items() if k not in keys]
    notes = []
    if skipped:
        notes.append(f"**Coverage:** this run covers {len(keys)} of the planned {len(COMPANIES)} companies due to "
                     f"budget. Skipped: {', '.join(skipped)}.")
    for key, source in reused.items():
        notes.append(f"**{COMPANIES[key]}:** pipeline state reused from `{source}`, not re-run. Its pipeline cost was "
                     "spent in that earlier run and isn't in the Cost column.")
    if any(UNRECORDED in r["metrics"]["providers_used"] for r in runs):
        notes.append(f"**Providers `{UNRECORDED}`:** the trace predates provider recording.")
    if any("nim" in r["metrics"]["providers_used"] for r in runs):
        notes.append("**NIM fallback:** claims any NIM answer touched are left out of every claim metric (count in "
                     "parentheses), and a run with any NIM answer has no report scores.")
    return (f"# Eval results ({name})\n\nPipeline model: {g.MODEL}. Judge: {judge.JUDGE_MODEL}. "
            f"One live run per company; web research varies between runs.\n\n"
            + "".join(f"{n}\n\n" for n in notes) + summary_table(runs) + "\n")


# --- Driver ------------------------------------------------------------------------------------------------------

def run_pipeline(company, posting):
    start = last = time.perf_counter()
    stages, final = Counter(), None
    for state in g.stream_auto_approved(g.initial_state(company, posting)):
        now = time.perf_counter()
        if state["trace"]:
            stages[state["trace"][-1]["node"]] += now - last
        last, final = now, state
    return final, {"total_s": round(time.perf_counter() - start, 1), **{k: round(v, 1) for k, v in stages.items()}}


def main(keys, reuse_dir=None):
    out = EVALS / "results" / datetime.now().strftime("%Y-%m-%d_%H%M")
    (out / "states").mkdir(parents=True)
    judge.meter_pipeline()
    runs, reused = [], {}
    for key in keys:
        company, posting = COMPANIES[key], (EVALS / "postings" / f"{key}.txt").read_text(encoding="utf-8")
        before = {"pipeline": judge.cost(g.MODEL), "judge": judge.cost(judge.JUDGE_MODEL)}
        saved = reuse_dir / f"{key}.json" if reuse_dir else None
        if saved and saved.exists():
            print(f"[{key}] reusing pipeline state from {saved}", flush=True)
            state, timing = json.loads(saved.read_text(encoding="utf-8")), None
            reused[key] = saved.relative_to(EVALS.parent).as_posix()
        else:
            print(f"[{key}] running pipeline...", flush=True)
            state, timing = run_pipeline(company, posting)
        (out / "states" / f"{key}.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{key}] judging...", flush=True)
        rows = score_claims(claim_rows(state, company, key), state, company)
        metrics = run_metrics(rows, state)
        # The briefing is written from every upstream answer, so any NIM answer makes it a mixed-provider report.
        report = (score_report(state) if "nim" not in metrics["providers_used"] else
                  {"report_faithfulness": None, "report_hallucination": None, "unverified_stated_as_fact": None})
        run = {"key": key, "company": company, "timing": timing, "metrics": metrics, "report": report,
               "planner": score_planner(state["trace"], company)}
        run["cost"] = {"pipeline": round(judge.cost(g.MODEL) - before["pipeline"], 3) if timing else None,
                       "judge": round(judge.cost(judge.JUDGE_MODEL) - before["judge"], 3)}
        runs.append(run)
        with open(out / "claims.jsonl", "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")
        # Rewritten after every company, so a run stopped partway (e.g. by the budget cap) still has a summary.
        (out / "summary.md").write_text(summary_md(out.name, runs, keys, reused), encoding="utf-8")
        print(f"[{key}] done, ${(run['cost']['pipeline'] or 0) + run['cost']['judge']:.2f} "
              f"(total so far ${judge.cost():.2f})", flush=True)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    reuse = None
    if "--reuse" in args:  # --reuse <states dir>: judge those saved pipeline states instead of re-running them
        i = args.index("--reuse")
        reuse, args = Path(args[i + 1]).resolve(), args[:i] + args[i + 2:]
    main(args or list(COMPANIES), reuse)
