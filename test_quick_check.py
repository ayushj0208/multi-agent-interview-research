"""Offline tests for the quick check (no API calls): Tavily and the model are scripted."""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
os.environ.setdefault("TAVILY_API_KEY", "tvly-dummy")

from typing import get_args  # noqa: E402

from pydantic import ValidationError  # noqa: E402

import graph as g  # noqa: E402
import quick_check as qc  # noqa: E402

LCA_URL = "https://h1bdata.info/index.php?em=CaseGuard"
VISA = [{"url": LCA_URL, "title": "CaseGuard H-1B salaries",
         "content": "CASEGUARD INC   filed 3 LCAs in\n2024 for Software Engineer roles in Arlington, VA."}]
CAREERS_URL = "https://caseguard.com/careers"
EVERIFY = [{"url": CAREERS_URL, "title": "Careers", "content": "CaseGuard is an equal opportunity employer."}]


class FakeTavily:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def search(self, query, **options):
        self.calls.append((query, options))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return {"results": response}


def run(answer, *responses):
    prompts = []
    g.tavily = FakeTavily(*(responses or (VISA, EVERIFY)))
    g.ask = lambda model, prompt, attempts=1: prompts.append(prompt) or (answer, "claude")
    return qc.quick_check("CaseGuard"), prompts, g.tavily.calls


def answer(h1b_verdict="couldnt_determine", h1b_evidence=(), everify_verdict="couldnt_determine", everify_evidence=()):
    return qc.QuickCheck(
        h1b=qc.VisaAnswer(verdict=h1b_verdict, summary="h1b summary", evidence=[qc.Evidence(url=u, quote=q) for u, q in h1b_evidence]),
        e_verify=qc.EVerifyAnswer(verdict=everify_verdict, summary="e-verify summary",
                                  evidence=[qc.Evidence(url=u, quote=q) for u, q in everify_evidence]))


def test_yes_survives_only_with_a_quote_found_in_the_result_it_cites():
    # Case and whitespace differ from the snippet, but the words are the same: kept.
    out, _, _ = run(answer("yes_with_evidence", [(LCA_URL, "CaseGuard Inc filed 3 LCAs in 2024")]))
    assert out["h1b"]["verdict"] == "yes_with_evidence" and len(out["h1b"]["evidence"]) == 1

    for evidence in ([(LCA_URL, "CaseGuard Inc filed 40 LCAs in 2024")],           # quote not in the snippet
                     [("https://myvisajobs.com/x", "filed 3 LCAs in 2024")],      # URL that wasn't a result
                     [(LCA_URL, "   ")],                                          # empty quote
                     []):                                                          # no evidence at all
        out, _, _ = run(answer("yes_with_evidence", evidence))
        assert out["h1b"]["verdict"] == "couldnt_determine" and out["h1b"]["evidence"] == [], evidence
        assert "none of its quoted evidence appears" in out["h1b"]["summary"]

    out, _, _ = run(answer(everify_verdict="yes_with_evidence", everify_evidence=[(CAREERS_URL, "participates in E-Verify")]))
    assert out["e_verify"]["verdict"] == "couldnt_determine", "an invented E-Verify statement must not pass"


def test_no_history_found_stands_without_a_quote():
    out, _, _ = run(answer("no_history_found"))
    assert out["h1b"]["verdict"] == "no_history_found"


def test_e_verify_answer_cannot_say_no():
    assert get_args(qc.EVerifyAnswer.model_fields["verdict"].annotation) == ("yes_with_evidence", "couldnt_determine")
    for verdict in ("no", "no_history_found"):
        try:
            qc.EVerifyAnswer(verdict=verdict, summary="s")
            raise AssertionError(f"E-Verify must not accept {verdict!r}")
        except ValidationError:
            pass
    _, prompts, _ = run(answer())
    assert "a missing statement is NOT evidence that they don't use it" in " ".join(prompts[0].split())


def test_failed_searches_are_unknown_and_skip_the_model_when_both_fail():
    out, prompts, _ = run(answer("yes_with_evidence", [(LCA_URL, "filed 3 LCAs")]),
                          RuntimeError("tavily down"), RuntimeError("tavily down"))
    assert prompts == [], "nothing to judge, so no paid call"
    assert out["h1b"]["verdict"] == out["e_verify"]["verdict"] == "couldnt_determine" and out["provider"] is None

    # Only the visa search failed: its answer is unknown even though the model said yes; E-Verify still judged.
    out, prompts, _ = run(answer("yes_with_evidence", [(LCA_URL, "filed 3 LCAs")], "yes_with_evidence",
                                 [(CAREERS_URL, "equal opportunity employer")]),
                          RuntimeError("tavily down"), EVERIFY)
    assert len(prompts) == 1 and out["h1b"]["verdict"] == "couldnt_determine"
    assert out["h1b"]["summary"] == "The search failed, so this couldn't be checked."
    assert out["e_verify"]["verdict"] == "yes_with_evidence"


def test_visa_search_is_limited_to_visa_data_sites():
    _, prompts, calls = run(answer())
    (visa_query, visa_options), (everify_query, everify_options) = calls
    assert '"CaseGuard"' in visa_query and "H-1B" in visa_query and visa_options["include_domains"] == qc.VISA_DATA_SITES
    assert "E-Verify" in everify_query and "include_domains" not in everify_options
    assert "filed 3 LCAs" in prompts[0] and "equal opportunity employer" in prompts[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
