"""Offline tests for the quick check (no API calls): Tavily and the model are scripted."""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
os.environ.setdefault("TAVILY_API_KEY", "tvly-dummy")

import graph as g  # noqa: E402
import quick_check as qc  # noqa: E402

LCA_URL = "https://h1bdata.info/index.php?em=CaseGuard"
VISA = [{"url": LCA_URL, "title": "CaseGuard H-1B salaries",
         "content": "CASEGUARD INC   filed 3 LCAs in\n2024 for Software Engineer roles in Arlington, VA."}]


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
    g.tavily = FakeTavily(*(responses or (VISA,)))
    g.ask = lambda model, prompt, attempts=1: prompts.append(prompt) or (answer, "claude")
    return qc.quick_check("CaseGuard"), prompts, g.tavily.calls


def answer(h1b_verdict="couldnt_determine", h1b_evidence=()):
    return qc.QuickCheck(h1b=qc.VisaAnswer(verdict=h1b_verdict, summary="h1b summary",
                                           evidence=[qc.Evidence(url=u, quote=q) for u, q in h1b_evidence]))


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


def test_no_history_found_stands_without_a_quote():
    out, _, _ = run(answer("no_history_found"))
    assert out["h1b"]["verdict"] == "no_history_found"


def test_a_failed_search_is_unknown_and_skips_the_model():
    out, prompts, _ = run(answer("yes_with_evidence", [(LCA_URL, "filed 3 LCAs")]), RuntimeError("tavily down"))
    assert prompts == [], "nothing to judge, so no paid call"
    assert out["h1b"]["verdict"] == "couldnt_determine" and out["provider"] is None
    assert out["h1b"]["summary"] == "The search failed, so this couldn't be checked."


def test_visa_search_is_limited_to_visa_data_sites():
    _, prompts, calls = run(answer())
    ((query, options),) = calls
    assert '"CaseGuard"' in query and "H-1B" in query and options["include_domains"] == qc.VISA_DATA_SITES
    assert "filed 3 LCAs" in prompts[0]


def test_the_quick_check_is_h1b_only():
    out, prompts, calls = run(answer())
    assert set(qc.QuickCheck.model_fields) == {"h1b"}
    assert set(out) == {"company_name", "provider", "visa_data_sites", "h1b"}, "the H-1B answer and nothing else"
    assert len(calls) == 1 and len(prompts) == 1, "one search on visa data sites, one model call"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
