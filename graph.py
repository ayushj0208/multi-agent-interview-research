import os
import re
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, TypedDict, get_args
from urllib.parse import urlparse

import anthropic
import openai
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, ValidationError
from tavily import TavilyClient

load_dotenv()

MODEL = "claude-sonnet-5"
NIM_MODEL = "nvidia/nemotron-3-super-120b-a12b"
# LOW_COST_MODE=1 (the public demo) runs shallower: 2 automatic Researcher passes and 5 verification questions a
# round, instead of 3 and 8. Local dev leaves it unset for full depth. Same model either way.
LOW_COST_MODE = os.getenv("LOW_COST_MODE") == "1"
MAX_RETRIES = 2 if LOW_COST_MODE else 3  # caps automatic Researcher passes, including Editor send-backs
QUALITY_THRESHOLD = 7  # editor score out of 10
# ponytail: bounds CoVe search cost; raise if key claims go unchecked
MAX_VERIFICATION_QUESTIONS = 5 if LOW_COST_MODE else 8
MIN_CLAIMS_PER_CATEGORY = 3  # a category with fewer cumulative claims than this is "thin" and gets researched again
# Each send-back costs another Researcher pass + Editor review in Claude spend; in low-cost mode one already eats
# most of the savings.
MAX_HUMAN_SEND_BACKS = 1 if LOW_COST_MODE else 2
# Rough wall-clock minutes, shown on the posting and progress screens (via GET /pipeline). Full depth is measured:
# the CaseGuard eval run took 253 s, 175 s of it before the review pause. Low-cost is estimated by scaling (2 of 3
# Researcher passes, about 2/3 of the claims to review), not yet measured on a live run.
EXPECTED_MINUTES = {"to_review": 2, "total": 3} if LOW_COST_MODE else {"to_review": 3, "total": 4}
# ponytail: word-overlap heuristics, tuned on the live Sony run. Posting-derived claims scored 71%+ against the posting,
# web-researched ones 58% or less. Search snippets that were copies of the posting (job boards, careers pages) scored
# 66-100%, other pages 20-48%. Swap for embeddings if real postings start landing near these cutoffs.
POSTING_MATCH = 0.65  # claim restates the pasted posting
MIRROR_MATCH = 0.60  # search snippet is itself a copy of the posting
SNIPPET_STATES_CLAIM = 0.50  # cited snippet actually contains the claim

Category = Literal["news", "funding", "tech_stack", "culture", "role_context"]
CATEGORIES = get_args(Category)

claude = anthropic.Anthropic()
# Nemotron's reasoning mode ignores the JSON schema and burns the token budget thinking; off, it honors the schema.
nim = (ChatOpenAI(model=NIM_MODEL, base_url="https://integrate.api.nvidia.com/v1",
                  api_key=os.environ["NVIDIA_API_KEY"], max_tokens=8192, max_retries=6,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
       if os.getenv("NVIDIA_API_KEY") else None)
tavily = TavilyClient()


class ResearchState(TypedDict):
    company_name: str
    job_posting: str
    raw_findings: list[dict]
    verified_findings: dict
    final_report: str
    iteration_count: int
    editor_feedback: str | None
    status: str
    trace: list[dict]


def ask_claude(response_model, prompt, attempts):
    # The API enforces the JSON schema; attempts only re-asks when a Pydantic constraint the schema
    # can't express (e.g. 3-5 talking points) fails client-side validation.
    for attempt in range(1, attempts + 1):
        try:
            # Streamed so max_tokens can exceed the SDK's non-streaming cap: Sonnet 5 thinks by default, and on a
            # long Editor review thinking alone took ~12K tokens, truncating the JSON at the old 16K limit.
            with claude.messages.stream(model=MODEL, max_tokens=32000, output_format=response_model,
                                        messages=[{"role": "user", "content": prompt}]) as stream:
                response = stream.get_final_message()
        except ValidationError:
            if attempt == attempts:
                raise
            continue
        if response.parsed_output is None:
            raise RuntimeError(f"Claude returned no structured output (stop_reason={response.stop_reason})")
        return response.parsed_output


def ask_nim(response_model, prompt, attempts):
    structured = nim.with_structured_output(response_model, method="json_schema", strict=True)
    # Re-ask on bad output: schema failure, or a runaway generation hitting max_tokens.
    return structured.with_retry(stop_after_attempt=attempts,
                                 retry_if_exception_type=(OutputParserException, openai.LengthFinishReasonError)).invoke(prompt)


BUDGET_MESSAGE = "The demo's budget for this month is used up, so it can't run new checks or briefings. Please try again next month."


class BudgetExhausted(Exception):
    """Claude refused for money (credits gone or spend cap reached) and there's no fallback: a visitor-facing stop."""


def out_of_budget(e: anthropic.APIStatusError):
    """Out of credits, or a spend limit set in the Anthropic Console was reached. Both arrive as a 400
    invalid_request_error, not a billing_error. The credit message is the live one we saw (2026-09-25); the
    usage-limit wording is matched loosely because we haven't seen that error live yet."""
    text = str(e).lower()
    return (e.type == "billing_error"
            or (e.type == "invalid_request_error" and ("credit balance is too low" in text or "usage limit" in text)))


def out_of_capacity(e: anthropic.APIStatusError):
    """Claude can't serve us right now: rate-limited, or out of budget."""
    return e.type == "rate_limit_error" or out_of_budget(e)


def ask(response_model, prompt, attempts=1):
    """Returns (answer, provider). The provider goes into the trace so NIM answers can be told apart downstream."""
    try:
        return ask_claude(response_model, prompt, attempts), "claude"
    except anthropic.APIStatusError as e:
        if nim is None or not out_of_capacity(e):
            if out_of_budget(e):  # the deployed demo runs without NIM: say so plainly instead of a raw API error
                raise BudgetExhausted(BUDGET_MESSAGE) from e
            raise
        print(f"    Claude {e.type}; retrying this call on NVIDIA NIM ({NIM_MODEL}). Anthropic said: {e}")
        # ponytail: Nemotron occasionally loops until max_tokens; one retry clears it. Raise if it recurs.
        return ask_nim(response_model, prompt, max(attempts, 2)), "nim"


def initial_state(company_name, job_posting) -> ResearchState:
    return {"company_name": company_name, "job_posting": job_posting, "raw_findings": [], "verified_findings": {},
            "final_report": "", "iteration_count": 0, "editor_feedback": None, "status": "Starting", "trace": []}


def progress(node, stage, iteration, message, **detail):
    """Emit a live-progress event to stream_mode="custom" consumers (the API's SSE feed)."""
    try:
        writer = get_stream_writer()
    except RuntimeError:  # called outside a graph run, e.g. a node invoked directly in a test
        return
    writer({"node": node, "stage": stage, "iteration": iteration, "message": message, "detail": detail})


def search(query):
    # ponytail: 3 results x 600 chars keeps prompts (and token cost) small; raise if briefings feel thin
    results = tavily.search(query, max_results=3)["results"]
    return [{"url": r["url"], "content": r["content"][:600]} for r in results]


# --- Researcher -------------------------------------------------------------

class Query(BaseModel):
    category: Category
    query: str


class QueryPlan(BaseModel):
    queries: list[Query]


class Finding(BaseModel):
    claim: str
    category: Category
    source_url: str


class ResearchPass(BaseModel):
    findings: list[Finding]


STOPWORDS = {"with", "that", "this", "from", "have", "their", "they", "your", "will", "into", "about", "more", "than",
             "also", "such", "which", "were", "been", "across", "including", "other"}


def words(text):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9+#./-]{3,}", text.lower()) if w not in STOPWORDS}


def overlap(text, reference_words):
    """Share of text's words that also appear in the reference."""
    w = words(text)
    return len(w & reference_words) / len(w) if w else 0.0


def attribute_source(finding, posting_words, snippets):
    """A fact from the pasted posting is sourced to the posting, not to whatever search result sat next to it.

    It keeps a web URL only when the cited page genuinely states the fact on its own, i.e. that page is not
    a job-board or careers-page copy of the same posting.
    """
    if overlap(finding["claim"], posting_words) < POSTING_MATCH:
        return finding
    snippet = snippets.get(finding["source_url"], "")
    independent = (overlap(finding["claim"], words(snippet)) >= SNIPPET_STATES_CLAIM
                   and overlap(snippet, posting_words) < MIRROR_MATCH)
    return finding if independent else {**finding, "source_url": None, "from_posting": True}


def thin_categories(findings):
    counts = Counter(f["category"] for f in findings)
    return [c for c in CATEGORIES if counts[c] < MIN_CLAIMS_PER_CATEGORY]


def researcher(state: ResearchState):
    iteration = state["iteration_count"] + 1
    gaps = thin_categories(state["raw_findings"]) if not state["editor_feedback"] else list(CATEGORIES)

    progress("researcher", "planning", iteration, f"Planning searches for: {', '.join(gaps)}", categories=gaps)
    plan, plan_by = ask(QueryPlan, f"""You are researching {state['company_name']} to prepare a candidate for an interview.
Job posting:
{state['job_posting']}

Categories to cover now: {gaps}
Editor feedback on the previous research (address it if present): {state['editor_feedback'] or 'none'}
Claims already found: {[f['claim'] for f in state['raw_findings']]}

Write one targeted web search query per category that needs work. Don't repeat ground already covered.""")

    found, snippets, extract_by = [], {}, None  # a pass that plans no queries has nothing new to extract
    if plan.queries:
        progress("researcher", "searching", iteration, f"Searching the web ({len(plan.queries)} queries)",
                 queries=[q.model_dump() for q in plan.queries])
        with ThreadPoolExecutor(max_workers=len(CATEGORIES)) as pool:
            hits = list(pool.map(search, [q.query for q in plan.queries]))
        results = [{"category": q.category, "query": q.query, "results": h} for q, h in zip(plan.queries, hits)]
        snippets = {r["url"]: r["content"] for h in hits for r in h}
        progress("researcher", "extracting", iteration, "Extracting claims from search results")
        extracted, extract_by = ask(ResearchPass, f"""Extract specific, factual claims about {state['company_name']} from these search results,
relevant to a candidate applying for this role:
{state['job_posting']}

Search results:
{results}

Extract at most 12 claims, the most useful for interview prep, one sentence each.
Tag every claim with its category and the exact source URL it came from. Only extract what the results actually say.""")
        found = extracted.findings

    # The extractor sees earlier claims (to judge gaps) and sometimes echoes them back; keep only new ones.
    # ponytail: exact-text match; paraphrased repeats still get through and are merged by the Editor.
    seen = {f["claim"].strip().lower() for f in state["raw_findings"]}
    posting_words = words(state["job_posting"])
    added = []
    for f in found:
        key = f.claim.strip().lower()
        if key not in seen:
            seen.add(key)
            added.append(attribute_source(f.model_dump(), posting_words, snippets))

    raw = state["raw_findings"] + added
    thin = thin_categories(raw)  # cumulative across all passes, decided in code rather than by the model
    status = (f"Researcher pass {iteration}: {len(plan.queries)} searches, {len(added)} new findings "
              f"({len(raw)} total), thin: {thin or 'none'}")
    return {
        "raw_findings": raw,
        "iteration_count": iteration,
        "editor_feedback": None,
        "status": status,
        "trace": state["trace"] + [{"node": "researcher", "iteration": iteration,
                                    "queries": [q.model_dump() for q in plan.queries], "new_findings": len(added),
                                    "total_findings": len(raw), "thin_categories": thin, "status": status,
                                    # "extract" produced this pass's new findings (the last new_findings of raw)
                                    "providers": {"plan": plan_by, "extract": extract_by}}],
    }


def route_research(state: ResearchState):
    if state["trace"][-1]["thin_categories"] and state["iteration_count"] < MAX_RETRIES:
        return "researcher"
    return "editor"


# --- Editor (Chain-of-Verification) ------------------------------------------

class VerificationQuestion(BaseModel):
    claim_index: int
    question: str


class VerificationPlan(BaseModel):
    questions: list[VerificationQuestion]


class Answer(BaseModel):
    answer: str
    source_url: str | None


class VerifiedClaim(BaseModel):
    claim_index: int = Field(description="the claim's index in the list above; for merged duplicates, the lowest index")
    claim: str
    category: Category
    status: Literal["verified", "unverified", "contradicted", "from_posting", "off_target"]
    confidence: Literal["high", "medium", "low"]
    citation_ok: bool = Field(description="false if the claim's source_url is generic or unrelated and doesn't "
                                          "actually support this specific claim")
    unconfirmed_specifics: list[str] = Field(
        default_factory=list, description="specifics the claim states (names, titles, dates, figures) that its evidence "
                                          "did not confirm: its own independent check, or for from_posting the job "
                                          "posting text; empty if the evidence confirmed all of them")
    conflicting_values: list[str] = Field(
        default_factory=list, description="for contradicted: values its own check states that are incompatible with the "
                                          "claim, e.g. 'check says 2019, claim says 2014'. A rounded or less precise "
                                          "version of the claim's figure (e.g. '30+' for '33') is not conflicting")
    note: str = ""


STRONG_LABELS = {"verified": "verified", "from_posting": "from the job posting", "contradicted": "contradicted"}


def enforce_status(c: VerifiedClaim, checked_indexes):
    """A strong label is earned in full or not at all: short of that the claim is downgraded to unverified, with why.

    verified: its own check confirmed every specific. from_posting: the posting states every specific.
    contradicted: its check states a genuinely conflicting value, not just a less precise one."""
    if c.status == "verified" and c.claim_index not in checked_indexes:
        reason = "no independent check targeted this claim itself"
    elif c.status == "verified" and c.unconfirmed_specifics:
        reason = f"its independent check didn't confirm: {'; '.join(c.unconfirmed_specifics)}"
    elif c.status == "from_posting" and c.unconfirmed_specifics:
        reason = f"the posting doesn't state: {'; '.join(c.unconfirmed_specifics)}"
    elif c.status == "contradicted" and not c.conflicting_values:
        reason = "its check found a less precise or partial version, not a conflicting value" + (
            f" ({'; '.join(c.unconfirmed_specifics)})" if c.unconfirmed_specifics else "")
    else:
        return c
    return c.model_copy(update={"status": "unverified",
                                "note": f"{c.note} Not {STRONG_LABELS[c.status]}: {reason}.".strip()})


# ponytail: hostname heuristics for picking among supporting sources; a domain-reputation list would do better.
LOW_QUALITY_HOSTS = ("crack", "torrent", "warez", "repack", "apk", "leak")  # never used as a citation
AGGREGATOR_HOSTS = ("rocketreach", "leadiq", "zoominfo", "glassdoor", "indeed", "builtin", "comparably", "crunchbase",
                    "tracxn", "getlatka", "linkedin", "youtube", "instagram", "reddit", "quora", "medium.com")
REPUTABLE_HOSTS = ("wikipedia.org", "reuters.com", "bloomberg.com", "cnbc.com", "ft.com", "wsj.com", "theverge.com",
                   "techcrunch.com", "apnews.com")


def source_quality(url, company_name):
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if any(m in host for m in LOW_QUALITY_HOSTS):
        return -2
    if any(a in host for a in AGGREGATOR_HOSTS):
        return -1
    company_tokens = [t for t in re.findall(r"[a-z0-9]{4,}", company_name.lower())]
    if any(t in host for t in company_tokens) or host.endswith((".gov", ".edu")) or host.endswith(REPUTABLE_HOSTS):
        return 2
    return 0


def best_supporting_source(claim, results, company_name):
    """Among search results, the best-quality one whose content actually states the claim; None if none does."""
    candidates = [(source_quality(r["url"], company_name), overlap(claim, words(r["content"])), r["url"])
                  for r in results]
    usable = [c for c in candidates if c[1] >= SNIPPET_STATES_CLAIM and c[0] > -2]
    return max(usable)[2] if usable else None


def cited_source(c: VerifiedClaim, raw_findings, checks, company_name):
    """The citation a reviewed claim keeps: its own if sound, else a good source that independently supports it."""
    original = raw_findings[c.claim_index]["source_url"] if 0 <= c.claim_index < len(raw_findings) else None
    if c.citation_ok or original is None:  # None: sourced to the pasted job posting, nothing to correct
        return original, c.note
    if c.status == "verified":
        results = [r for k in checks if k["claim_index"] == c.claim_index for r in k["results"]]
        if better := best_supporting_source(c.claim, results, company_name):
            return better, f"{c.note} Citation corrected to a source that states this (was {original}).".strip()
    # A known-bad citation is dropped rather than shown as if it supported the claim.
    return None, f"{c.note} Original citation ({original}) doesn't support this claim.".strip()


class EditorReview(BaseModel):
    claims: list[VerifiedClaim]
    quality_score: int = Field(ge=1, le=10)
    feedback: str = Field(description="specific, actionable gaps for the researcher to fix")


def editor(state: ResearchState):
    claims = [f"[{i}] {f['claim']}" for i, f in enumerate(state["raw_findings"])]
    iteration = state["iteration_count"]

    # Step 1: questions see only the claim text.
    progress("editor", "questioning", iteration, f"Choosing key claims to verify out of {len(claims)}")
    plan, questions_by = ask(VerificationPlan, f"""Here are claims about {state['company_name']}:
{claims}

Pick the {MAX_VERIFICATION_QUESTIONS} most important factual claims (numbers, dates, funding, products, leadership, tech)
and write one standalone search question per claim that would independently confirm or refute it.
Don't restate the claim's answer inside the question.""")
    questions = plan.questions[:MAX_VERIFICATION_QUESTIONS]

    # Step 2: each question gets its own fresh search and its own answering call, run concurrently. A call sees
    # only its question and that question's results: never the Researcher's draft, never another question's data.
    progress("editor", "verifying", iteration, f"Verifying {len(questions)} claims with fresh searches",
             questions=[q.question for q in questions])

    def verify(q):
        results = search(q.question)
        answer, answered_by = ask(Answer, f"""Answer this question using only the search results below. If they don't answer it, say so.
Question: {q.question}
Search results: {results}""")
        # results are kept (for citation correction) but left out of the review prompt below to keep it small.
        return {"claim_index": q.claim_index, "question": q.question, **answer.model_dump(), "results": results,
                "provider": answered_by}

    with ThreadPoolExecutor(max_workers=MAX_VERIFICATION_QUESTIONS) as pool:
        checks = list(pool.map(verify, questions))

    # Step 3: compare independent answers against the original claims, judging relevance against the posting.
    progress("editor", "reviewing", iteration,
             f"Cross-checking {len(state['raw_findings'])} claims against {len(checks)} independent answers",
             claims=len(state["raw_findings"]), checks=[{"question": c["question"], "answer": c["answer"]} for c in checks])
    review, review_by = ask(EditorReview, f"""You are the editor for a research briefing on {state['company_name']}, for a candidate applying to this role:
{state['job_posting']}

Researcher's claims (with sources; source_url null means the claim was taken from the job posting above):
{[{'index': i, **f} for i, f in enumerate(state['raw_findings'])]}

Independent verification (answered from fresh searches, without seeing the claims):
{[{k: v for k, v in c.items() if k not in ("results", "provider")} for c in checks]}

Give every claim exactly one status:
- off_target: the fact itself is about a different company, product or entity than {state['company_name']} (a same-name
  company, a parent, sibling or subsidiary, another product line), even if it's correct. Check this first, and judge the
  entity from the source_url as well as the claim text: a claim that names {state['company_name']} but whose data comes
  from a page about another entity (e.g. a profile or tech-stack page for a sibling company) is off_target.
- from_posting: only if the job posting text above actually states it (responsibilities, requirements, team, tech used
  in the role). These need no web verification. If the posting doesn't say it, use one of the other statuses. If the
  posting states only part of it, list every specific the claim adds beyond the posting (a name, product, figure) in
  unconfirmed_specifics, and the claim will not count as from_posting.
- verified: this claim's OWN independent check (the check whose claim_index is this claim's index) confirms every
  specific the claim states: names, titles, dates, figures. A check aimed at a different claim never counts, even a
  similar one. If its own check confirms only part of the claim, list what it didn't confirm in unconfirmed_specifics
  (e.g. "the blog post title") and the claim will not count as verified.
- contradicted: the independent answer states a value incompatible with the claim (a different number, date, name); list
  those values in conflicting_values. A rounded or less precise version of the claim's figure (e.g. "30+" for "33",
  "over 100" for "120") is compatible, not a contradiction: use unverified and put the discrepancy in
  unconfirmed_specifics.
- unverified: not checked, or the answer was inconclusive; judge confidence from the source quality.

Citations are judged separately from status. If a claim is about {state['company_name']} but its source_url is generic
or unrelated (e.g. a job-search or reviews page that isn't about this company or doesn't contain the fact), set
citation_ok to false and still give it the status its content earns. A bad citation alone is never a reason for
off_target, and never a reason to downgrade a claim the independent answer confirms.

Give each claim its claim_index from the list above. Then merge duplicates, score overall quality 1-10 (coverage of news, funding, tech stack, culture and role context,
and how many key claims held up), and write specific feedback naming missing, contradicted or off-target items
the researcher should fix.""")

    send_back = review.quality_score < QUALITY_THRESHOLD and state["iteration_count"] < MAX_RETRIES
    checked_indexes = {k["claim_index"] for k in checks}
    reviewed = [enforce_status(c, checked_indexes) for c in review.claims]
    verified = {}
    for c in reviewed:
        source_url, note = cited_source(c, state["raw_findings"], checks, state["company_name"])
        verified.setdefault(c.category, []).append(
            {**c.model_dump(exclude={"category", "citation_ok"}), "source_url": source_url, "note": note})
    counts = {s: n for s in get_args(VerifiedClaim.model_fields["status"].annotation)
              if (n := sum(c.status == s for c in reviewed))}
    status = (f"Editor checked {len(checks)} claims independently: {counts}, quality {review.quality_score}/10"
              + (" - sending back to Researcher" if send_back else " - passing to Writer"))
    return {
        "verified_findings": verified,
        "editor_feedback": review.feedback if send_back else None,
        "status": status,
        "trace": state["trace"] + [{"node": "editor", "iteration": iteration, "checks": checks,
                                    "quality_score": review.quality_score, "counts": counts,
                                    "decision": "send_back" if send_back else "pass",
                                    "feedback": review.feedback, "status": status,
                                    "providers": {"questions": questions_by, "review": review_by}}],
    }


def route_editor(state: ResearchState):
    return "researcher" if state["editor_feedback"] else "review"


# --- Human review --------------------------------------------------------------

def human_send_backs(trace):
    return sum(t["node"] == "review" and t["action"] == "send_back" for t in trace)


def review(state: ResearchState):
    # interrupt() must stay first: on resume LangGraph re-runs this node from the top and interrupt()
    # returns the human's decision instead of pausing, so nothing before it may have side effects.
    editor_entry = next(t for t in reversed(state["trace"]) if t["node"] == "editor")
    decision = interrupt({"verified_findings": state["verified_findings"],
                          "quality_score": editor_entry["quality_score"], "counts": editor_entry["counts"],
                          "feedback": editor_entry["feedback"], "pass_threshold": QUALITY_THRESHOLD,
                          # Below the threshold, the Editor would have sent it back; only the pass cap stopped it.
                          "passes_exhausted": editor_entry["quality_score"] < QUALITY_THRESHOLD,
                          "send_backs_left": MAX_HUMAN_SEND_BACKS - human_send_backs(state["trace"])})
    send_back = decision["action"] == "send_back"
    status = f"Human reviewer {'sent the research back' if send_back else 'approved the findings'}"
    return {
        "editor_feedback": f"Human reviewer: {decision['feedback']}" if send_back else None,
        "status": status,
        "trace": state["trace"] + [{"node": "review", "iteration": state["iteration_count"],
                                    "action": decision["action"], "feedback": decision.get("feedback", ""),
                                    "status": status}],
    }


def route_review(state: ResearchState):
    return "researcher" if state["editor_feedback"] else "writer"


# --- Writer ------------------------------------------------------------------

class Point(BaseModel):
    text: str = Field(description="the point itself, with no citation text")
    source_url: str | None = Field(description="the finding's source_url copied exactly, or null")
    from_posting: bool = Field(description="true if the point comes from the job posting the candidate provided")


POSTING_SOURCE = "the job posting you provided"


class Section(BaseModel):
    title: str
    points: list[Point]


class Briefing(BaseModel):
    intro: str
    sections: list[Section]
    talking_points: list[str] = Field(min_length=3, max_length=5)
    questions_to_ask: list[str] = Field(min_length=3)


def cite(p: Point):
    if p.source_url:
        return f" ([source]({p.source_url}))"
    return f" (from {POSTING_SOURCE})" if p.from_posting else ""


def to_markdown(company, b: Briefing):
    lines = [f"# {company}: Interview Briefing", "", b.intro, ""]
    for s in b.sections:
        lines += [f"## {s.title}", *[f"- {p.text}{cite(p)}" for p in s.points], ""]
    lines += ["## Talking Points", *[f"{i}. {p}" for i, p in enumerate(b.talking_points, 1)], ""]
    lines += ["## Questions to Ask Them", *[f"- {q}" for q in b.questions_to_ask], ""]
    points = [p for s in b.sections for p in s.points]
    sources = list(dict.fromkeys(p.source_url for p in points if p.source_url))
    if any(p.from_posting and not p.source_url for p in points):
        sources.insert(0, POSTING_SOURCE[0].upper() + POSTING_SOURCE[1:])
    lines += ["## Sources", *[f"- {u}" for u in sources]]
    return "\n".join(lines)


def writer(state: ResearchState):
    usable = {cat: [c for c in claims if c["status"] != "off_target"]
              for cat, claims in state["verified_findings"].items()}
    progress("writer", "drafting", state["iteration_count"], "Drafting the briefing")
    briefing, briefing_by = ask(Briefing, f"""Write an interview-prep briefing on {state['company_name']} for a candidate applying to this role:
{state['job_posting']}

Use only these editor-reviewed findings:
{usable}

Rules: state "verified" claims plainly; attribute "from_posting" ones to the job posting; never present "contradicted"
claims as fact. Make "unverified" claims clearly tentative, but word it the way a careful analyst would and vary it from
point to point: "reportedly", "one source says ... though this couldn't be confirmed", "unconfirmed reports suggest",
"according to <source>, not yet independently verified". Never open a bullet with a fixed label like "This is unverified:".
Sections: one per research category that has findings (Recent News, Funding & Business, Tech Stack, Culture, Role Context).
Each point's text carries no citation; put the finding's source_url in the point's source_url field. A finding with
status "from_posting" came from the job posting the candidate provided: set that point's from_posting to true and leave
source_url null, and never cite a job board or search page for it. Any other finding with a null source_url has no
usable source: leave source_url null and from_posting false.
Talking points: 3-5, each tying a company fact to the candidate's fit for the role.
Questions to ask: thoughtful questions the candidate can ask the interviewers.""", attempts=3)

    return {
        "final_report": to_markdown(state["company_name"], briefing),
        "status": f"Writer finished: {len(briefing.sections)} sections, {len(briefing.talking_points)} talking points",
        "trace": state["trace"] + [{"node": "writer", "iteration": state["iteration_count"],
                                    "sections": len(briefing.sections), "briefing": briefing.model_dump(),
                                    "provider": briefing_by}],
    }


builder = StateGraph(ResearchState)
builder.add_node("researcher", researcher)
builder.add_node("editor", editor)
builder.add_node("review", review)
builder.add_node("writer", writer)
builder.add_edge(START, "researcher")
builder.add_conditional_edges("researcher", route_research, ["researcher", "editor"])
builder.add_conditional_edges("editor", route_editor, ["researcher", "review"])
builder.add_conditional_edges("review", route_review, ["researcher", "writer"])
builder.add_edge("writer", END)
graph = builder.compile(checkpointer=InMemorySaver())  # interrupt() needs a checkpointer, even in-process


def stream_auto_approved(state):
    """Run to completion with no human: approve at the review pause. For the CLI and evals."""
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    graph_input = state
    while True:
        for values in graph.stream(graph_input, config, stream_mode="values"):
            if "__interrupt__" not in values:
                yield values
        if not graph.get_state(config).interrupts:
            return
        graph_input = Command(resume={"action": "approve", "feedback": ""})
