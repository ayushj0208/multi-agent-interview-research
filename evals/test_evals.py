"""Offline checks for the eval's own logic (no API calls). Run: python evals/test_evals.py"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
os.environ.setdefault("TAVILY_API_KEY", "tvly-dummy")

import run_evals as ev  # noqa: E402

COMPANY = "Sony Play Station"


def claim(i, status, url="https://en.wikipedia.org/wiki/x", unconfirmed=(), note=""):
    return {"claim_index": i, "claim": f"claim {i}", "status": status, "source_url": url,
            "unconfirmed_specifics": list(unconfirmed), "note": note, "confidence": "high"}


STATE = {
    "job_posting": "posting text",
    "verified_findings": {"news": [
        claim(0, "verified", note="Citation corrected to a source that states this (was x)."),
        claim(1, "unverified"),                                    # never checked: a budget miss candidate
        claim(2, "unverified", unconfirmed=["the blog post title"]),  # checked, partially confirmed
        claim(3, "unverified"),                                    # checked, inconclusive
        claim(4, "from_posting", url=None),
        claim(5, "contradicted", url="https://rocketreach.co/x", note="Original citation (x) doesn't support this claim.")]},
    "trace": [
        {"node": "researcher", "iteration": 1, "thin_categories": ["culture"], "new_findings": 4,
         "providers": {"plan": "claude", "extract": "claude"},
         "queries": [{"category": c, "query": f"Sony PlayStation {c}"} for c in ("news", "funding", "tech_stack")]},
        {"node": "researcher", "iteration": 2, "thin_categories": [], "new_findings": 2,
         "providers": {"plan": "claude", "extract": "claude"},
         "queries": [{"category": "culture", "query": "culture at the company"}]},
        {"node": "editor", "decision": "send_back", "checks": [], "providers": {"questions": "claude", "review": "claude"}},
        {"node": "researcher", "iteration": 3, "thin_categories": [], "queries": [], "new_findings": 0,
         "providers": {"plan": "claude", "extract": None}},
        {"node": "editor", "decision": "pass", "providers": {"questions": "claude", "review": "claude"},
         "checks": [{"claim_index": i, "question": "q", "answer": "a", "results": [], "provider": "claude"}
                    for i in (0, 2, 3, 5)]},
        {"node": "writer", "provider": "claude"}],
}


def with_nim_check(state, claim_index):
    """The same run, but claim_index's own check fell back to NIM."""
    trace = [dict(t) for t in state["trace"]]
    trace[4]["checks"] = [{**k, "provider": "nim" if k["claim_index"] == claim_index else k["provider"]}
                          for k in trace[4]["checks"]]
    return {**state, "trace": trace}


def test_claim_rows_bucket_unverified_by_failure_mode():
    rows = {r["claim_index"]: r for r in ev.claim_rows(STATE, COMPANY, "sony")}
    assert rows[1]["bucket"] == "never_checked" and not rows[1]["own_check"]
    assert rows[2]["bucket"] == "partial" and rows[3]["bucket"] == "inconclusive"
    assert rows[0]["bucket"] is None and rows[0]["own_check"] and rows[0]["citation_corrected"]
    assert rows[4]["source_type"] == "posting" and rows[5]["source_type"] == "aggregator" and rows[5]["citation_dropped"]
    assert rows[0]["source_type"] == "company_or_reputable"


def test_recheck_selection_covers_every_failure_mode_and_samples_budget_misses():
    rows = ev.claim_rows(STATE, COMPANY, "sony")
    assert {r["claim_index"] for r in ev.select_for_recheck(rows)} == {0, 1, 2, 3, 5}  # from_posting judged separately


def test_attribution_scoped_claims_are_detected():
    assert ev.attribution_scoped("SIE has approximately 8.3K employees according to LeadIQ's employee metrics.")
    assert ev.attribution_scoped("RocketReach lists SIE HR department staff")
    assert not ev.attribution_scoped("Ghost of Yotei launched on PlayStation 5 on October 2, 2025.")


def test_planner_gaps_follow_the_code_decided_list_and_reset_after_send_back():
    samples = ev.planner_samples(STATE["trace"])
    assert [gaps for gaps, _, _ in samples] == [list(ev.g.CATEGORIES), ["culture"], list(ev.g.CATEGORIES)]
    assert ev.tool_call_f1(["culture"], ["culture"]) == 1.0
    assert round(ev.tool_call_f1(list(ev.g.CATEGORIES), ["news", "funding", "tech_stack"]), 2) == 0.75  # 3 of 5
    assert ev.tool_call_f1(list(ev.g.CATEGORIES), []) == 0.0  # a pass that searched nothing while gaps existed
    planner = ev.score_planner(STATE["trace"], COMPANY)
    assert planner["queries_naming_company"] == 0.75  # "culture at the company" doesn't name it


def test_run_metrics_report_precision_as_counts_not_rates_of_verified():
    rows = ev.claim_rows(STATE, COMPANY, "sony")
    verdicts = {0: "supported", 1: "supported", 2: "unconfirmable", 3: "supported", 5: "contradicted"}
    for r in rows:
        if r["claim_index"] in verdicts:
            r["recheck_verdict"] = verdicts[r["claim_index"]]
    m = ev.run_metrics(rows, STATE)
    assert m["verified_precision"] == {"k": 1, "n": 1}
    assert m["budget_miss"] == {"k": 1, "n": 1}  # the never-checked claim was confirmable: a budget miss
    assert m["truly_unconfirmable"] == {"k": 1, "n": 2}  # partial one unconfirmable, inconclusive one wasn't
    assert m["contradicted_agreement"] == {"k": 1, "n": 1}
    assert m["from_posting_precision"] == {"k": 0, "n": 0}  # not judged in this fixture
    assert m["editor_send_backs"] == 1 and m["researcher_passes"] == 3
    assert "verified_rate" not in m  # deliberately not a metric: more "verified" isn't better


def test_nim_touched_claims_are_labeled_and_left_out_of_every_score():
    state = with_nim_check(STATE, 0)  # the only verified claim was checked by NIM
    rows = {r["claim_index"]: r for r in ev.claim_rows(state, COMPANY, "sony")}
    assert rows[0]["nim_involved"] and rows[0]["providers"] == ["claude", "nim"]
    assert not any(rows[i]["nim_involved"] for i in (1, 2, 3, 4, 5))
    assert 0 not in {r["claim_index"] for r in ev.select_for_recheck(list(rows.values()))}, "don't pay to judge it"
    for r in rows.values():
        r["recheck_verdict"] = "supported"
    m = ev.run_metrics(list(rows.values()), state)
    assert m["verified_precision"] == {"k": 0, "n": 0} and m["claims_excluded_nim"] == 1
    assert m["providers_used"] == ["claude", "nim"]


def test_nim_planned_pass_is_left_out_of_planner_score():
    trace = [dict(t) for t in STATE["trace"]]
    trace[1] = {**trace[1], "providers": {"plan": "nim", "extract": "nim"}}
    assert [s[1] for s in ev.planner_samples(trace)] == [["news", "funding", "tech_stack"], []]


def test_traces_from_before_provider_recording_are_unrecorded_not_claude():
    legacy = {**STATE, "trace": [{k: v for k, v in t.items() if k not in ("providers", "provider")} |
                                 ({"checks": [{kk: vv for kk, vv in c.items() if kk != "provider"} for c in t["checks"]]}
                                  if "checks" in t else {}) for t in STATE["trace"]]}
    rows = ev.claim_rows(legacy, COMPANY, "sony")
    assert all(r["providers"] == [ev.UNRECORDED] and not r["nim_involved"] for r in rows)
    assert ev.run_providers(legacy["trace"]) == [ev.UNRECORDED]


def test_summary_table_renders_counts():
    rows = ev.claim_rows(STATE, COMPANY, "sony")
    for r in rows:
        r["recheck_verdict"] = "supported"
    run = {"company": COMPANY, "metrics": ev.run_metrics(rows, STATE), "timing": {"total_s": 240.0},
           "report": {"report_faithfulness": 0.9, "report_hallucination": 0.1, "unverified_stated_as_fact": 2},
           "planner": {"planner_tool_call_f1": 0.8}, "cost": {"pipeline": 0.3, "judge": 1.1}}
    table = ev.summary_table([run])
    assert "| Sony Play Station | 1 / 3 / 1 / 1 / 0 | 1/1 |" in table and "| claude (0) | 240s | $1.40 |" in table
    assert table.splitlines()[1].count("---") == table.splitlines()[0].count("|") - 1

    reused = {**run, "timing": None, "cost": {"pipeline": None, "judge": 0.9},
              "report": {"report_faithfulness": None, "report_hallucination": None, "unverified_stated_as_fact": None}}
    assert "| n/a | n/a | n/a | 0.80 | claude (0) | reused | $0.90 (judge only) |" in ev.summary_table([reused])

    md = ev.summary_md("x", [reused], ["sony_play_station", "caseguard", "notion"],
                       {"sony_play_station": "evals/results/2026-09-25_1218/states/sony_play_station.json"})
    assert "covers 3 of the planned 6 companies due to budget. Skipped: Scout AI, Stripe, Databricks." in md
    assert "reused from `evals/results/2026-09-25_1218/states/sony_play_station.json`, not re-run" in md


def test_l3_keeps_false_and_unestablished_apart():
    import limitations
    original = limitations.judge.recheck
    limitations.judge.recheck = lambda claim, company: (type("V", (), {"verdict": "supported", "reasoning": ""})(), [])
    try:
        l3 = limitations.l3_source_vs_fact()
    finally:
        limitations.judge.recheck = original
    by_fact = {r["id"]: r["true_as_fact"] for r in l3["rows"]}
    assert l3["judge_supported_but_false"] == [i for i, t in by_fact.items() if t is False] == ["b4f105cc#10"]
    assert l3["judge_supported_but_unestablished"] == [i for i, t in by_fact.items() if t is None] == ["52156394#8"]
    assert l3["not_true_as_fact"] == 1 and l3["unestablished"] == 1 and l3["true_as_fact"] == 8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
