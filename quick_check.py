"""Quick check before the full briefing, from the company name alone: has it sponsored work visas, and is it a
known E-Verify employer? Two Tavily searches and one Claude call, about $0.02 against a full briefing's $0.40+.

It never guesses. A "yes" must quote, word for word, a search result it cites; otherwise code downgrades it to
"couldn't determine". E-Verify has no "no" answer at all: participation isn't reliably public, so missing evidence
proves nothing either way.
"""
import re
from typing import Literal

from pydantic import BaseModel, Field

import graph as g

# USCIS's H-1B Employer Data Hub, plus sites republishing the Department of Labor's LCA disclosure data (the DOL
# files themselves are bulk spreadsheets that web search doesn't index).
VISA_DATA_SITES = ["uscis.gov", "h1bdata.info", "myvisajobs.com", "h1bgrader.com"]
RESULTS = 5
SNIPPET_CHARS = 1500


class Evidence(BaseModel):
    url: str = Field(description="a URL copied exactly from the search results")
    quote: str = Field(description="a short passage copied word for word from that result's content")


class VisaAnswer(BaseModel):
    verdict: Literal["yes_with_evidence", "no_history_found", "couldnt_determine"]
    summary: str = Field(description="one sentence")
    evidence: list[Evidence] = Field(default_factory=list)


class EVerifyAnswer(BaseModel):
    verdict: Literal["yes_with_evidence", "couldnt_determine"]  # deliberately no "no": see the module docstring
    summary: str = Field(description="one sentence")
    evidence: list[Evidence] = Field(default_factory=list)


class QuickCheck(BaseModel):
    h1b: VisaAnswer
    e_verify: EVerifyAnswer


def search(query, **options):
    """Returns (results, failed). A failed search becomes "couldn't determine", never a guess."""
    try:
        results = g.tavily.search(query, max_results=RESULTS, **options)["results"]
    except Exception:  # any Tavily or network failure: report unknown rather than break the screen
        return [], True
    return [{"url": r["url"], "title": r.get("title", ""), "content": r["content"][:SNIPPET_CHARS]} for r in results], False


def normalized(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def enforce(answer, results, search_failed):
    """Keep only evidence that is really in the cited result; a "yes" left with none is downgraded to unknown."""
    if search_failed:
        return answer.model_copy(update={"verdict": "couldnt_determine", "evidence": [],
                                         "summary": "The search failed, so this couldn't be checked."})
    content = {r["url"]: normalized(r["content"]) for r in results}
    kept = [e for e in answer.evidence if e.url in content and normalized(e.quote) and normalized(e.quote) in content[e.url]]
    if answer.verdict == "yes_with_evidence" and not kept:
        return answer.model_copy(update={
            "verdict": "couldnt_determine", "evidence": [],
            "summary": "The model answered yes, but none of its quoted evidence appears in the search results it "
                       "cited, so this is treated as unknown."})
    return answer.model_copy(update={"evidence": kept})


def unknown(model):
    return model(verdict="couldnt_determine", summary="The search failed, so this couldn't be checked.")


def quick_check(company):
    visa, visa_failed = search(f'"{company}" H-1B LCA visa sponsorship', include_domains=VISA_DATA_SITES)
    everify, everify_failed = search(f'"{company}" E-Verify employer')
    provider = None
    if visa_failed and everify_failed:  # nothing to judge: skip the paid call
        answer = QuickCheck(h1b=unknown(VisaAnswer), e_verify=unknown(EVerifyAnswer))
    else:
        answer, provider = g.ask(QuickCheck, f"""You are checking two facts about the employer "{company}" for a job seeker.
Use ONLY the search results below, never your own knowledge. When unsure, answer couldnt_determine: a wrong yes or
no is worse than an honest "couldn't determine".

1. H-1B / work-visa sponsorship history. Results from visa-disclosure sites that republish US Department of Labor LCA
and USCIS H-1B data:
{visa}
- yes_with_evidence: a result shows this employer filed LCAs or H-1B petitions (counts, years or job titles). The
  employer name must match "{company}", not a different company with a similar name; say which name and years the
  data shows.
- no_history_found: the results are visa-data pages and none of them show filings by this employer.
- couldnt_determine: the results are missing or unclear, or could be a different company with a similar name.

2. E-Verify participation. General web results:
{everify}
- yes_with_evidence: a result explicitly states that this employer participates in E-Verify (for example its own
  careers page or a job posting says so).
- couldnt_determine: anything else. E-Verify participation is not reliably public, so a missing statement is NOT
  evidence that they don't use it. Never imply that they don't.

For each, write a one-sentence summary. For any yes_with_evidence, give evidence: the exact URL from the results and a
short passage copied word for word from that result's content.""")
    return {"company_name": company, "provider": provider, "visa_data_sites": VISA_DATA_SITES,
            "h1b": enforce(answer.h1b, visa, visa_failed).model_dump(),
            "e_verify": enforce(answer.e_verify, everify, everify_failed).model_dump()}
