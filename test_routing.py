import os

os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
os.environ.setdefault("TAVILY_API_KEY", "tvly-dummy")

import contextlib
import threading

import anthropic
import httpx
from pydantic import ValidationError

import graph as g
import quick_check as qc
from graph import MAX_RETRIES, route_editor, route_research

real_ask, real_ask_claude = g.ask, g.ask_claude


CREDITS_EXHAUSTED = ("Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to "
                     "upgrade or purchase credits.")


def anthropic_error(cls, status, error_type, message="simulated"):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    body = {"type": "error", "error": {"type": error_type, "message": message}}
    return cls(f"Error code: {status} - {body}", response=response, body=body)  # the SDK's own message format


def use_fakes(claude_exc, calls, nim=object()):
    def ask_claude(response_model, prompt, attempts):
        calls.append("claude")
        raise claude_exc

    def ask_nim(response_model, prompt, attempts):
        calls.append("nim")
        return "answer"
    g.ask_claude, g.ask_nim, g.nim = ask_claude, ask_nim, nim


def test_rate_limit_and_billing_fall_back_to_nim():
    for exc in (anthropic_error(anthropic.RateLimitError, 429, "rate_limit_error"),
                anthropic_error(anthropic.APIStatusError, 402, "billing_error"),
                # What Anthropic actually returned when the balance ran out (live, 2026-09-25).
                anthropic_error(anthropic.BadRequestError, 400, "invalid_request_error", CREDITS_EXHAUSTED)):
        calls = []
        use_fakes(exc, calls)
        assert real_ask(g.Answer, "q") == ("answer", "nim"), "a NIM answer must be labeled as NIM"
        assert calls == ["claude", "nim"]


def test_other_errors_and_missing_nim_do_not_fall_back():
    for exc in (anthropic_error(anthropic.BadRequestError, 400, "invalid_request_error"),
                anthropic_error(anthropic.InternalServerError, 529, "overloaded_error")):
        calls = []
        use_fakes(exc, calls)
        try:
            real_ask(g.Answer, "q")
            raise AssertionError(f"{exc.type} should propagate")
        except anthropic.APIStatusError:
            assert calls == ["claude"]

    use_fakes(anthropic_error(anthropic.RateLimitError, 429, "rate_limit_error"), [], nim=None)
    try:
        real_ask(g.Answer, "q")
        raise AssertionError("with no NVIDIA key the rate limit should propagate")
    except anthropic.RateLimitError:
        pass


SPEND_CAP_REACHED = ("You have reached your specified workspace API usage limits. You will regain access on "
                     "2026-11-01 at 00:00 UTC.")  # assumed wording: not yet seen live


def test_without_nim_an_exhausted_budget_is_a_clear_stop_not_a_raw_error():
    for exc in (anthropic_error(anthropic.BadRequestError, 400, "invalid_request_error", CREDITS_EXHAUSTED),
                anthropic_error(anthropic.BadRequestError, 400, "invalid_request_error", SPEND_CAP_REACHED),
                anthropic_error(anthropic.APIStatusError, 402, "billing_error")):
        use_fakes(exc, [], nim=None)  # the deployed demo: no NVIDIA key
        try:
            real_ask(g.Answer, "q")
            raise AssertionError("should have stopped")
        except g.BudgetExhausted as stop:
            assert str(stop) == g.BUDGET_MESSAGE and stop.__cause__ is exc

    for exc in (anthropic_error(anthropic.BadRequestError, 400, "invalid_request_error"),  # not about money
                anthropic_error(anthropic.RateLimitError, 429, "rate_limit_error")):     # busy, not broke
        use_fakes(exc, [], nim=None)
        try:
            real_ask(g.Answer, "q")
            raise AssertionError("should have raised")
        except g.BudgetExhausted:
            raise AssertionError(f"{exc.type} is not a budget problem")
        except anthropic.APIStatusError:
            pass


def test_claude_re_asks_only_on_validation_failure():
    class FakeMessages:
        def __init__(self, failures):
            self.failures, self.calls = failures, 0

        def stream(self, **kwargs):
            self.calls += 1
            return contextlib.nullcontext(self)

        def get_final_message(self):
            if self.calls <= self.failures:
                g.Briefing.model_validate({})  # raises the same ValidationError a constraint failure would
            return type("Response", (), {"parsed_output": "briefing", "stop_reason": "end_turn"})()

    g.claude = type("Client", (), {})()
    g.claude.messages = FakeMessages(failures=2)
    assert real_ask_claude(g.Briefing, "p", attempts=3) == "briefing" and g.claude.messages.calls == 3

    g.claude.messages = FakeMessages(failures=1)
    try:
        real_ask_claude(g.Answer, "p", attempts=1)
        raise AssertionError("a single attempt should surface the validation error")
    except ValidationError:
        assert g.claude.messages.calls == 1


LOW_COST_PROBE = """
import os, sys
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy"); os.environ.setdefault("TAVILY_API_KEY", "tvly-dummy")
import graph as g
passes, questions, send_backs, to_review, total = map(int, sys.argv[1:6])
settings = (g.MAX_RETRIES, g.MAX_VERIFICATION_QUESTIONS, g.MAX_HUMAN_SEND_BACKS, g.EXPECTED_MINUTES)
assert settings == (passes, questions, send_backs, {"to_review": to_review, "total": total}), settings

thin = {"trace": [{"thin_categories": ["funding"]}]}
assert g.route_research({**thin, "iteration_count": passes - 1}) == "researcher"
assert g.route_research({**thin, "iteration_count": passes}) == "editor", "no Researcher pass beyond the cap"

prompts = {}
eight = [g.VerificationQuestion(claim_index=i, question=f"Q{i}?") for i in range(8)]
def ask(model, prompt, attempts=1):
    prompts[model] = prompt
    if model is g.VerificationPlan:
        return g.VerificationPlan(questions=eight), "claude"
    if model is g.Answer:
        return g.Answer(answer="a", source_url=None), "claude"
    return g.EditorReview(claims=[], quality_score=4, feedback="thin"), "claude"
g.ask, g.search = ask, (lambda q: [])
out = g.editor({"company_name": "X", "job_posting": "p", "raw_findings": [{"claim": f"c{i}", "category": "news",
                "source_url": "https://ex.com"} for i in range(10)], "verified_findings": {}, "final_report": "",
                "iteration_count": passes, "editor_feedback": None, "status": "", "trace": []})
assert f"Pick the {questions} most important" in prompts[g.VerificationPlan]
assert len(out["trace"][-1]["checks"]) == questions, "extra questions from the model are cut to the budget"
assert out["trace"][-1]["decision"] == "pass", "at the pass cap a low score goes to the human, not back"
"""


def test_low_cost_mode_is_shallower_and_off_by_default():
    import subprocess
    import sys
    # (LOW_COST_MODE, Researcher passes, questions, human send-backs, minutes to review, minutes in total)
    for env_value, *expected in ((None, 3, 8, 2, 3, 4), ("1", 2, 5, 1, 2, 3), ("0", 3, 8, 2, 3, 4)):
        env = {k: v for k, v in os.environ.items() if k != "LOW_COST_MODE"}
        if env_value is not None:
            env["LOW_COST_MODE"] = env_value
        done = subprocess.run([sys.executable, "-c", LOW_COST_PROBE, *map(str, expected)], env=env,
                              capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        assert done.returncode == 0, f"LOW_COST_MODE={env_value}: {done.stderr[-600:]}"


def test_research_loops_while_thin_then_halts_at_cap():
    thin = {"trace": [{"thin_categories": ["funding"]}]}
    assert route_research({**thin, "iteration_count": 1}) == "researcher"
    assert route_research({**thin, "iteration_count": MAX_RETRIES}) == "editor"
    assert route_research({"trace": [{"thin_categories": []}], "iteration_count": 1}) == "editor"


def test_editor_sends_back_only_with_feedback():
    assert route_editor({"editor_feedback": "funding figures contradicted"}) == "researcher"
    assert route_editor({"editor_feedback": None}) == "review"


POSTING = "Software Engineer I on the PlayStation application framework team. Uses React Native and C++."
DRAFT_CLAIM = "SIE sold 303.3 million game copies in FY2024"
LCA_URL = "https://h1bdata.info/index.php?em=Sony+Interactive+Entertainment"
QUICK_CHECK_RESULTS = [{"url": LCA_URL, "title": "Sony Interactive Entertainment H-1B",
                        "content": "SONY INTERACTIVE ENTERTAINMENT LLC filed 212 LCAs in 2024, mostly software engineers."}]
QUICK_CHECK = qc.QuickCheck(
    h1b=qc.VisaAnswer(verdict="yes_with_evidence", summary="Sony Interactive Entertainment LLC filed 212 LCAs in 2024.",
                      evidence=[qc.Evidence(url=LCA_URL, quote="filed 212 LCAs in 2024")]),
    e_verify=qc.EVerifyAnswer(verdict="couldnt_determine", summary="No result states whether it uses E-Verify."))


def fake_ask(prompts, quality_score=8):
    canned = {
        g.QueryPlan: g.QueryPlan(queries=[g.Query(category="news", query="SIE news")]),
        g.ResearchPass: g.ResearchPass(findings=[
            g.Finding(claim=DRAFT_CLAIM, category="funding", source_url="https://ex.com/sales"),
            g.Finding(claim="Sony New Media Solutions uses Route 53", category="tech_stack", source_url="https://ex.com/snms"),
            g.Finding(claim="The role works in React Native and C++", category="role_context", source_url="https://ex.com/job")]),
        g.VerificationPlan: g.VerificationPlan(questions=[g.VerificationQuestion(claim_index=0, question="How many games did SIE sell in FY2024?")]),
        g.Answer: g.Answer(answer="303.3 million", source_url="https://ex.com/ir"),
        g.EditorReview: g.EditorReview(quality_score=quality_score, claims=[
            g.VerifiedClaim(claim_index=0, claim=DRAFT_CLAIM, category="funding", status="verified", confidence="high", citation_ok=True),
            g.VerifiedClaim(claim_index=1, claim="Sony New Media Solutions uses Route 53", category="tech_stack", status="off_target", confidence="high", citation_ok=True),
            g.VerifiedClaim(claim_index=2, claim="The role works in React Native and C++", category="role_context", status="from_posting", confidence="high", citation_ok=True)],
            # Two dollar amounts on one line: unescaped, Streamlit markdown renders the span between them as LaTeX.
            feedback="Pay range is $182K-289K; FY25 revenue $31.09 billion with operating income of $3.07 billion."),
        g.Briefing: g.Briefing(intro="Intro.", talking_points=["a", "b", "c"], questions_to_ask=["x", "y", "z"], sections=[
            g.Section(title="Funding & Business", points=[g.Point(text="Sold 303.3M copies.", source_url="https://ex.com/sales", from_posting=False),
                                                          g.Point(text="Revenue $31.09 billion, operating income $3.07 billion.", source_url=None, from_posting=False)]),
            g.Section(title="Role Context", points=[g.Point(text="The role works in React Native and C++.", source_url=None, from_posting=True)])]),
        qc.QuickCheck: QUICK_CHECK,
    }

    def ask(model, prompt, attempts=1):
        prompts[model] = prompt
        return canned[model], "claude"
    return ask


def claude_only(fake):
    """Adapts a scripted LLM that returns bare answers to ask()'s (answer, provider) shape."""
    return lambda model, prompt, attempts=1: (fake(model, prompt, attempts), "claude")


def test_pipeline_offline():
    prompts = {}
    g.ask, g.search = fake_ask(prompts), lambda q: [{"url": "https://ex.com/r", "content": "result"}]
    *_, final = g.stream_auto_approved(g.initial_state("Sony Interactive Entertainment", POSTING))

    assert DRAFT_CLAIM not in prompts[g.Answer], "independent answers must never see the Researcher's draft"
    assert POSTING in prompts[g.EditorReview] and "off_target" in prompts[g.EditorReview]
    assert "Sony New Media Solutions" not in prompts[g.Briefing], "off-target claims must not reach the Writer"
    assert "from_posting" in prompts[g.Briefing]

    report = final["final_report"]
    assert "(source)" not in report
    assert "- Sold 303.3M copies. ([source](https://ex.com/sales))" in report
    assert report.endswith("## Sources\n- The job posting you provided\n- https://ex.com/sales")
    assert "- The role works in React Native and C++. (from the job posting you provided)" in report
    editor_status = next(t["status"] for t in final["trace"] if t["node"] == "editor")
    assert "'off_target': 1" in editor_status and "'from_posting': 1" in editor_status
    # The CLI path paused at human review and auto-approved exactly once before the Writer ran.
    assert [t["node"] for t in final["trace"]][-2:] == ["review", "writer"]
    assert [t["action"] for t in final["trace"] if t["node"] == "review"] == ["approve"]
    # Every LLM answer is labeled with its provider in the trace, but the label never reaches a prompt.
    by_node = {t["node"]: t for t in final["trace"]}
    assert by_node["researcher"]["providers"] == {"plan": "claude", "extract": "claude"}
    assert by_node["editor"]["providers"] == {"questions": "claude", "review": "claude"}
    assert all(k["provider"] == "claude" for k in by_node["editor"]["checks"]) and by_node["writer"]["provider"] == "claude"
    assert "provider" not in prompts[g.EditorReview]


def researcher_state(raw):
    return {"company_name": "Scout AI", "job_posting": "Full Stack Engineer", "raw_findings": raw,
            "verified_findings": {}, "final_report": "", "iteration_count": 1, "editor_feedback": None,
            "status": "", "trace": [{"node": "researcher", "thin_categories": ["culture"]}]}


def test_researcher_does_not_re_add_claims():
    old = {"claim": "Scout AI raised a $15M seed round", "category": "funding", "source_url": "https://ex.com/a"}
    searched = []
    g.search = lambda q: searched.append(q) or []

    responses = {g.QueryPlan: g.QueryPlan(queries=[])}
    g.ask = claude_only(lambda model, prompt, attempts=1: responses[model])
    out = g.researcher(researcher_state([old]))
    assert out["raw_findings"] == [old] and searched == []
    assert out["trace"][-1]["thin_categories"] == list(g.CATEGORIES)  # 1 funding claim is still under the threshold

    responses = {
        g.QueryPlan: g.QueryPlan(queries=[g.Query(category="culture", query="Scout AI culture")]),
        g.ResearchPass: g.ResearchPass(findings=[
            g.Finding(claim="  scout ai raised a $15M seed round ", category="funding", source_url="https://ex.com/a"),
            g.Finding(claim="Scout AI values in-person work", category="culture", source_url="https://ex.com/b"),
            g.Finding(claim="Scout AI values in-person work", category="culture", source_url="https://ex.com/c")]),
    }
    out = g.researcher(researcher_state([old]))
    assert [f["claim"] for f in out["raw_findings"]] == [old["claim"], "Scout AI values in-person work"]
    assert "1 new findings (2 total)" in out["status"]


def test_researcher_caps_searches_and_claims_in_code_not_just_the_prompt():
    # What a hostile posting could talk the planner and extractor into: 40 searches and 30 claims in one pass.
    plan = g.QueryPlan(queries=[g.Query(category="news", query=f"query {i}") for i in range(40)])
    extracted = g.ResearchPass(findings=[g.Finding(claim=f"Scout AI fact {i}", category="news",
                                                   source_url="https://ex.com/f") for i in range(30)])
    searched = []
    g.search = lambda q: searched.append(q) or []
    g.ask = claude_only(lambda model, prompt, attempts=1: {g.QueryPlan: plan, g.ResearchPass: extracted}[model])
    out = g.researcher(researcher_state([]))
    assert len(searched) == len(g.CATEGORIES) == len(out["trace"][-1]["queries"])
    assert out["trace"][-1]["new_findings"] == g.MAX_FINDINGS_PER_PASS == 12


def claims(category, n, tag=""):
    return [{"claim": f"{category} fact {tag}{i}", "category": category, "source_url": "https://ex.com"} for i in range(n)]


def test_thin_categories_from_counts():
    findings = claims("news", 7) + claims("funding", 7) + claims("tech_stack", g.MIN_CLAIMS_PER_CATEGORY)
    thin = g.thin_categories(findings)
    assert "news" not in thin and "funding" not in thin and "tech_stack" not in thin
    assert "culture" in thin and "role_context" in thin
    assert g.thin_categories([]) == list(g.CATEGORIES)


def test_thin_uses_cumulative_counts_not_current_pass():
    # The Scout AI pass-3 flip: earlier passes found 7 news and 7 funding claims, this pass only searched
    # culture/tech_stack. The old model-judged list flagged news and funding thin; the count must not.
    earlier = claims("news", 7) + claims("funding", 7) + claims("role_context", 10) + claims("tech_stack", 1)
    this_pass = [g.Finding(**c) for c in claims("culture", 6, "new") + claims("tech_stack", 2, "new")]
    responses = {g.QueryPlan: g.QueryPlan(queries=[g.Query(category="culture", query="culture"),
                                                   g.Query(category="tech_stack", query="tech")]),
                 g.ResearchPass: g.ResearchPass(findings=this_pass)}
    g.ask, g.search = claude_only(lambda model, prompt, attempts=1: responses[model]), (lambda q: [])
    out = g.researcher(researcher_state(earlier))
    assert out["trace"][-1]["thin_categories"] == []  # news 7, funding 7, role 10, tech 3, culture 6
    assert route_research({**out, "trace": out["trace"]}) == "editor"


def test_each_verification_call_sees_only_its_own_evidence():
    questions = [g.VerificationQuestion(claim_index=i, question=f"QUESTION-{i}?") for i in range(8)]
    draft = [{"claim": f"DRAFT-CLAIM-{i}", "category": "news", "source_url": f"https://ex.com/{i}"} for i in range(8)]
    all_searching = threading.Barrier(8, timeout=5)  # only opens if all 8 searches are in flight at once
    answer_prompts, lock = [], threading.Lock()

    def search(query):
        all_searching.wait()
        return [{"url": f"https://ex.com/{query}", "content": f"EVIDENCE-FOR-{query}"}]

    def ask(model, prompt, attempts=1):
        if model is g.VerificationPlan:
            return g.VerificationPlan(questions=questions)
        if model is g.Answer:
            with lock:
                answer_prompts.append(prompt)
            return g.Answer(answer=f"answered {prompt.split('Question: ')[1].split(chr(10))[0]}", source_url=None)
        return g.EditorReview(claims=[], quality_score=8, feedback="ok")

    g.ask, g.search = claude_only(ask), search
    out = g.editor({"company_name": "Sony Interactive Entertainment", "job_posting": "posting", "raw_findings": draft,
                    "verified_findings": {}, "final_report": "", "iteration_count": 1, "editor_feedback": None,
                    "status": "", "trace": []})

    assert len(answer_prompts) == 8, "one answering call per question, not one batched call"
    for prompt in answer_prompts:
        own = prompt.split("Question: ")[1].split("\n")[0]
        assert f"EVIDENCE-FOR-{own}" in prompt
        others = [q.question for q in questions if q.question != own]
        assert not any(o in prompt or f"EVIDENCE-FOR-{o}" in prompt for o in others), "saw another question's data"
        assert "DRAFT-CLAIM" not in prompt, "saw the Researcher's draft"
    checks = out["trace"][-1]["checks"]
    assert [c["claim_index"] for c in checks] == list(range(8))  # results line up with their questions
    assert all(c["answer"] == f"answered {c['question']}" for c in checks)


GLASSDOOR = "https://www.indeed.com/cmp/Glassdoor/reviews?ftopic=wlbalance"
POSTING_URL = "https://careers.playstation.com/software-engineer-i/job/6104600004"


def run_editor_on(raw, checks, review_claims, company="Sony Play Station",
                  posting="Estimated base pay range $137,300 - $205,900 USD."):
    """Run the real Editor node with a scripted LLM and search.

    checks: (claim_index, question, Answer, search results) per verification question.
    """
    prompts = {}
    answers = {question: answer for _, question, answer, _ in checks}
    results = {question: hits for _, question, _, hits in checks}

    def ask(model, prompt, attempts=1):
        prompts[model] = prompt
        if model is g.VerificationPlan:
            return g.VerificationPlan(questions=[g.VerificationQuestion(claim_index=i, question=q) for i, q, _, _ in checks])
        if model is g.Answer:
            return answers[prompt.split("Question: ")[1].split("\n")[0]]
        return g.EditorReview(claims=review_claims, quality_score=6, feedback="fix citations")

    g.ask, g.search = claude_only(ask), (lambda q: results[q])
    out = g.editor({"company_name": company, "job_posting": posting,
                    "raw_findings": raw, "verified_findings": {}, "final_report": "", "iteration_count": 3,
                    "editor_feedback": None, "status": "", "trace": []})
    return {c["claim_index"]: c for cs in out["verified_findings"].values() for c in cs}, prompts


def test_confirmed_claim_with_bad_citation_keeps_verified_and_gets_the_confirming_source():
    # The live "Sony Play Station" case: the pay range cited a generic Indeed/Glassdoor reviews page.
    raw = [{"claim": "The estimated base pay range for this role is $137,300 - $205,900 USD.", "category": "funding",
            "source_url": GLASSDOOR}]
    confirmed = g.Answer(answer="Yes, SIE lists a base pay range of $137,300 - $205,900 for this role.",
                         source_url=POSTING_URL)
    hits = [{"url": POSTING_URL, "content": "Software Engineer I, San Mateo. The estimated base pay range for this role "
                                            "is $137,300 - $205,900 USD."}]
    claims, prompts = run_editor_on(
        raw, [(0, "What is SIE's base pay range for Software Engineer I?", confirmed, hits)],
        [g.VerifiedClaim(claim_index=0, claim=raw[0]["claim"], category="funding", status="verified",
                         confidence="high", citation_ok=False, note="Accurate, but cited to a Glassdoor reviews page.")])

    pay = claims[0]
    assert pay["status"] == "verified", "a bad citation must not downgrade a confirmed claim"
    assert pay["source_url"] == POSTING_URL, "citation must be replaced by a source that states the claim"
    assert GLASSDOOR in pay["note"] and "corrected" in pay["note"]
    assert "A bad citation alone is never a reason for off_target" in " ".join(prompts[g.EditorReview].split())


YOTEI = "Ghost of Yōtei launched on PlayStation 5 on October 2, 2025."
CRACK = "https://crackrelease.com/ghost-of-yotei-confirmed-playstation-5-exclusive-launch"
NOTEBOOKCHECK = ("https://www.notebookcheck.net/Sony-officially-reveals-Ghost-of-Yotei-launch-date-and-pre-order-bonuses"
                 ".1003901.0.html")


def yotei_claim():
    return g.VerifiedClaim(claim_index=0, claim=YOTEI, category="news", status="verified", confidence="high",
                           citation_ok=False, note="Cites the generic press-release index, not the article.")


def test_correction_prefers_a_legitimate_supporting_source_over_a_low_quality_one():
    # The live case: both results state the launch date; the crack site came first and was picked.
    raw = [{"claim": YOTEI, "category": "news", "source_url": "https://sonyinteractive.com/en/news/press-releases"}]
    hits = [{"url": CRACK, "content": "Ghost of Yotei launched on PlayStation 5 on October 2, 2025 - download the "
                                      "full repack now, confirmed PS5 exclusive launch."},
            {"url": NOTEBOOKCHECK, "content": "Sony officially reveals Ghost of Yōtei launch date: the game launched on "
                                              "PlayStation 5 on October 2, 2025, with pre-order bonuses."}]
    answer = g.Answer(answer="Ghost of Yōtei launched on PlayStation 5 on October 2, 2025.", source_url=CRACK)
    claims, _ = run_editor_on(raw, [(0, "When did Ghost of Yōtei launch on PS5?", answer, hits)], [yotei_claim()])

    assert claims[0]["status"] == "verified"
    assert claims[0]["source_url"] == NOTEBOOKCHECK, "must prefer the legitimate source over the crack site"


def test_correction_drops_the_citation_when_no_result_supports_the_claim():
    raw = [{"claim": YOTEI, "category": "news", "source_url": "https://sonyinteractive.com/en/news/press-releases"}]
    hits = [{"url": NOTEBOOKCHECK, "content": "Best gaming laptops of 2025: our top picks for performance and battery."},
            {"url": "https://en.wikipedia.org/wiki/PlayStation_5", "content": "The PlayStation 5 is a home video game "
                                                                               "console developed by Sony."}]
    answer = g.Answer(answer="Ghost of Yōtei launched on PS5 on October 2, 2025.", source_url=NOTEBOOKCHECK)
    claims, _ = run_editor_on(raw, [(0, "When did Ghost of Yōtei launch on PS5?", answer, hits)], [yotei_claim()])

    assert claims[0]["status"] == "verified", "the claim's status is untouched; only the citation is dropped"
    assert claims[0]["source_url"] is None, "must not point at a result that doesn't state the claim"
    assert "doesn't support this claim" in claims[0]["note"]


FAMILY_BLOG = ("SIE published a blog post on 'Empowering Families with Easy and Customizable Family Play Experiences' "
               "introducing the PlayStation Family App on September 10, 2025.")
FAMILY_APP = ("Sony launched the PlayStation Family app in September 2025, a new mobile app giving parents an easy way "
              "to manage their children's gaming.")
FAMILY_URL = ("https://blog.playstation.com/2025/09/10/announcing-playstation-family-app-for-parental-controls-and-"
              "family-management")
LAUNCH_ANSWER = g.Answer(answer="Sony launched the PlayStation Family app on September 10, 2025, for iOS 14+ and "
                                "Android 8+.", source_url=FAMILY_URL)
LAUNCH_HITS = [{"url": FAMILY_URL, "content": "Sony launched the PlayStation Family app in September 2025, a new mobile "
                                              "app giving parents an easy way to manage their children's gaming."}]


def test_near_duplicate_cannot_borrow_its_twins_check():
    # The live #4/#24 case: only #24 was checked; the Editor still marked #4 verified using #24's answer.
    raw = [{"claim": FAMILY_BLOG, "category": "news", "source_url": "https://sonyinteractive.com/en/news/blog/category/product"},
           {"claim": FAMILY_APP, "category": "news", "source_url": FAMILY_URL}]
    claims, prompts = run_editor_on(raw, [(1, "When did Sony launch the PlayStation Family app?", LAUNCH_ANSWER, LAUNCH_HITS)], [
        g.VerifiedClaim(claim_index=0, claim=FAMILY_BLOG, category="news", status="verified", confidence="high",
                        citation_ok=False, note="Independent check confirms the app launched Sept 10, 2025."),
        g.VerifiedClaim(claim_index=1, claim=FAMILY_APP, category="news", status="verified", confidence="high",
                        citation_ok=True)])

    assert claims[0]["status"] == "unverified", "a claim can't be verified by a check aimed at another claim"
    assert "no independent check targeted this claim itself" in claims[0]["note"]
    assert claims[1]["status"] == "verified" and claims[1]["source_url"] == FAMILY_URL
    assert "A check aimed at a different claim never counts" in " ".join(prompts[g.EditorReview].split())


def test_partial_confirmation_of_a_compound_claim_is_not_verified():
    # No duplicate: the claim's own check confirms the launch date but not the blog post title it also claims.
    raw = [{"claim": FAMILY_BLOG, "category": "news", "source_url": "https://sonyinteractive.com/en/news/blog/category/product"}]
    claims, prompts = run_editor_on(raw, [(0, "When did Sony launch the PlayStation Family app?", LAUNCH_ANSWER, LAUNCH_HITS)], [
        g.VerifiedClaim(claim_index=0, claim=FAMILY_BLOG, category="news", status="verified", confidence="high",
                        citation_ok=False, unconfirmed_specifics=["the blog post title"],
                        note="Launch date of Sept 10, 2025 confirmed.")])

    assert claims[0]["status"] == "unverified", "confirming one specific of a compound claim isn't verified"
    assert "Launch date of Sept 10, 2025 confirmed." in claims[0]["note"]  # what was confirmed...
    assert "didn't confirm: the blog post title" in claims[0]["note"]  # ...and what wasn't
    assert claims[0]["source_url"] is None, "no longer verified, so no corrected citation either"
    assert "confirms every specific the claim states" in " ".join(prompts[g.EditorReview].split())


def test_bad_citation_without_confirmation_is_dropped_not_filtered():
    # What actually happened live: posting content with a made-up Glassdoor citation, never independently checked.
    raw = [{"claim": "SIE is a Fair Chance employer.", "category": "culture", "source_url": GLASSDOOR},
           {"claim": "Sony Music Entertainment has a 4.1 rating.", "category": "culture",
            "source_url": "https://www.indeed.com/cmp/Sony-Music-Entertainment/reviews"}]
    claims, _ = run_editor_on(raw, [], [
        g.VerifiedClaim(claim_index=0, claim=raw[0]["claim"], category="culture", status="from_posting",
                        confidence="high", citation_ok=False),
        g.VerifiedClaim(claim_index=1, claim=raw[1]["claim"], category="culture", status="off_target",
                        confidence="high", citation_ok=True)])

    assert claims[0]["status"] == "from_posting" and claims[0]["source_url"] is None
    assert GLASSDOOR in claims[0]["note"]
    assert claims[1]["status"] == "off_target" and claims[1]["source_url"].endswith("Sony-Music-Entertainment/reviews")


SONY_POSTING = open("samples/sony.txt", encoding="utf-8").read()
MIRROR = "https://www.wearedevelopers.com/jobs/ext/3027576-software-engineer"
NEWS = "https://www.gamesindustry.biz/sony-hiring-policy"


def test_posting_facts_are_sourced_to_the_posting_not_a_nearby_search_result():
    framework = next(line for line in SONY_POSTING.splitlines() if line.startswith("Our team develops"))
    snippets = {  # what Tavily returned next to these claims, 600-char snippets like search() produces
        MIRROR: framework[:600],  # a job-board copy of the same posting
        GLASSDOOR: "Glassdoor reviews: see what employees say about work-life balance, culture and compensation "
                   "at thousands of companies. Browse ratings and anonymous reviews.",
        NEWS: "Sony Interactive Entertainment conducts background checks at the offer stage for all new employees, "
              "a spokesperson told reporters, describing it as standard across its PlayStation studios worldwide.",
        "https://sonyinteractive.com/en/press-releases/2025/sie-partners-with-bad-robot-games":
            "Sony Interactive Entertainment partnered with Bad Robot Games to publish the studio's first game.",
    }
    findings = [
        g.Finding(claim="The framework combines React Native, native platform integration, reusable UI components, "
                        "SDK tooling, and developer documentation.", category="tech_stack", source_url=MIRROR),
        g.Finding(claim="Sony Interactive Entertainment is a Fair Chance employer and qualified applicants with arrest "
                        "and conviction records will be considered for employment.", category="culture",
                  source_url=GLASSDOOR),
        g.Finding(claim="Sony Interactive Entertainment conducts background checks at the offer stage for all new "
                        "employees.", category="culture", source_url=NEWS),
        g.Finding(claim="Sony Interactive Entertainment partnered with Bad Robot Games to publish the studio's first "
                        "game.", category="news",
                  source_url="https://sonyinteractive.com/en/press-releases/2025/sie-partners-with-bad-robot-games"),
    ]
    responses = {g.QueryPlan: g.QueryPlan(queries=[g.Query(category="culture", query="q")]),
                 g.ResearchPass: g.ResearchPass(findings=findings)}
    g.ask = claude_only(lambda model, prompt, attempts=1: responses[model])
    g.search = lambda q: [{"url": url, "content": text} for url, text in snippets.items()]
    state = {**researcher_state([]), "company_name": "Sony Interactive Entertainment", "job_posting": SONY_POSTING}
    by_claim = {f["claim"]: f for f in g.researcher(state)["raw_findings"]}

    mirrored, fair_chance, background, bad_robot = (by_claim[f.claim] for f in findings)
    assert mirrored["source_url"] is None and mirrored["from_posting"], "must not keep the job-board mirror's link"
    assert fair_chance["source_url"] is None and fair_chance["from_posting"], "must not keep the Glassdoor page"
    assert background["source_url"] == NEWS, "a genuine outside source stating the posting fact keeps its citation"
    assert bad_robot["source_url"].startswith("https://sonyinteractive.com") and "from_posting" not in bad_robot


def test_report_attributes_posting_points_to_the_posting():
    b = g.Briefing(intro="i", talking_points=["a", "b", "c"], questions_to_ask=["x", "y", "z"], sections=[
        g.Section(title="Role Context", points=[
            g.Point(text="The pay range is $137,300 - $205,900.", source_url=None, from_posting=True),
            g.Point(text="SIE partnered with Bad Robot Games.", source_url=NEWS, from_posting=False)])])
    report = g.to_markdown("Sony Interactive Entertainment", b)
    assert "- The pay range is $137,300 - $205,900. (from the job posting you provided)" in report
    assert report.endswith(f"## Sources\n- The job posting you provided\n- {NEWS}")


CASEGUARD_33 = ("CaseGuard Studio is an on-premise AI redaction software that redacts sensitive information across video, "
                "audio, documents, and images, and can automatically detect and redact 33 categories of PII, PCI, and PHI "
                "(also described elsewhere as covering documents and audio specifically).")
CASEGUARD_33_CHECK = (
    2, "What categories and number of PII, PCI, and PHI types can CaseGuard Studio automatically detect and redact?",
    g.Answer(answer="Based on the search results, CaseGuard Studio can automatically detect and redact 30+ categories of "
                    "PII, PCI, and PHI combined (as stated across its document redaction, call center, and general use "
                    "cases). Additionally, for PHI specifically, CaseGuard's AI-powered redaction can identify and redact "
                    "over 50 types of PHI, including patient names, diagnoses, addresses, and Social Security numbers.",
             source_url="https://caseguard.com/articles/document-redaction"),
    [{"url": "https://caseguard.com/articles/document-redaction", "content": "redact 30+ categories of PII, PCI, and PHI"}])


def test_less_precise_figure_is_not_a_contradiction():
    # Eval run 2026-09-25_1341, CaseGuard #2: claim says "33 categories", its check found "30+". The Editor called it
    # contradicted; the judge found caseguard.com stating 33 exactly. "30+" is compatible with 33, just less precise.
    raw = [{"claim": f"filler {i}", "category": "news", "source_url": "https://ex.com"} for i in range(2)] + [
        {"claim": CASEGUARD_33, "category": "tech_stack", "source_url": "https://caseguard.com/how-it-works"}]
    imprecise = g.VerifiedClaim(claim_index=2, claim=CASEGUARD_33, category="tech_stack", status="contradicted",
                                confidence="medium", citation_ok=True,
                                unconfirmed_specifics=["33 categories (the check found '30+ categories')"])
    by_index, prompts = run_editor_on(raw, [CASEGUARD_33_CHECK], [imprecise], company="CaseGuard")
    assert by_index[2]["status"] == "unverified"
    assert "Not contradicted: its check found a less precise or partial version" in by_index[2]["note"]
    assert "'30+ categories'" in by_index[2]["note"]
    assert '"30+" for "33"' in " ".join(prompts[g.EditorReview].split()), "the Editor must be told the rule itself"

    conflicting = imprecise.model_copy(update={"conflicting_values": ["check says 12 categories, claim says 33"],
                                               "unconfirmed_specifics": []})
    by_index, _ = run_editor_on(raw, [CASEGUARD_33_CHECK], [conflicting], company="CaseGuard")
    assert by_index[2]["status"] == "contradicted", "a genuinely conflicting value still contradicts"


SONY_100M = ("Sony Interactive Entertainment delivers hardware and network services (PlayStation Network, PlayStation "
             "Store, PlayStation Plus) to more than 100 million people as an entertainment leader.")
SONY_100M_POSTING = ("SIE is a dynamic technology company, delivering cutting-edge hardware and network services to more "
                     "than 100 million people and an entertainment leader, home to some of the most beloved and "
                     "recognizable intellectual properties (IP) in the world.")


def test_from_posting_requires_every_specific_in_the_posting():
    # Eval run 2026-09-25_1341, Sony #10: the posting says "hardware and network services to more than 100 million
    # people", but the claim adds three product names the posting never mentions. Enough wording matched to pass.
    raw = [{"claim": SONY_100M, "category": "role_context", "source_url": None, "from_posting": True}]
    added = g.VerifiedClaim(claim_index=0, claim=SONY_100M, category="role_context", status="from_posting",
                            confidence="high", citation_ok=True,
                            unconfirmed_specifics=["PlayStation Network", "PlayStation Store", "PlayStation Plus"])
    by_index, prompts = run_editor_on(raw, [], [added], posting=SONY_100M_POSTING)
    assert by_index[0]["status"] == "unverified" and by_index[0]["source_url"] is None
    assert by_index[0]["note"] == ("Not from the job posting: the posting doesn't state: PlayStation Network; "
                                   "PlayStation Store; PlayStation Plus.")
    assert "list every specific the claim adds beyond the posting" in " ".join(prompts[g.EditorReview].split())

    exact = added.model_copy(update={"unconfirmed_specifics": []})
    assert run_editor_on(raw, [], [exact], posting=SONY_100M_POSTING)[0][0]["status"] == "from_posting"


def test_writer_is_told_to_vary_unverified_wording():
    prompts = {}
    g.ask = fake_ask(prompts)
    g.writer({**g.initial_state("Sony Interactive Entertainment", POSTING),
              "verified_findings": {"news": [{"claim_index": 0, "claim": "c", "status": "unverified",
                                              "confidence": "low", "source_url": NEWS, "note": ""}]}})
    prompt = " ".join(prompts[g.Briefing].split())
    assert 'Never open a bullet with a fixed label like "This is unverified:"' in prompt
    assert "vary it from point to point" in prompt
    # A claim downgraded out of from_posting keeps its null source_url; only the status may credit the posting.
    assert "Any other finding with a null source_url has no usable source: leave source_url null and from_posting false" in prompt


if __name__ == "__main__":
    test_research_loops_while_thin_then_halts_at_cap()
    test_editor_sends_back_only_with_feedback()
    test_rate_limit_and_billing_fall_back_to_nim()
    test_other_errors_and_missing_nim_do_not_fall_back()
    test_claude_re_asks_only_on_validation_failure()
    test_pipeline_offline()
    test_researcher_does_not_re_add_claims()
    test_thin_categories_from_counts()
    test_thin_uses_cumulative_counts_not_current_pass()
    test_each_verification_call_sees_only_its_own_evidence()
    test_confirmed_claim_with_bad_citation_keeps_verified_and_gets_the_confirming_source()
    test_correction_prefers_a_legitimate_supporting_source_over_a_low_quality_one()
    test_correction_drops_the_citation_when_no_result_supports_the_claim()
    test_near_duplicate_cannot_borrow_its_twins_check()
    test_partial_confirmation_of_a_compound_claim_is_not_verified()
    test_bad_citation_without_confirmation_is_dropped_not_filtered()
    test_posting_facts_are_sourced_to_the_posting_not_a_nearby_search_result()
    test_report_attributes_posting_points_to_the_posting()
    test_writer_is_told_to_vary_unverified_wording()
    print("ok")
