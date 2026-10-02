"""Independent judge for the evals: a different, stronger model than the pipeline, on fresh evidence."""
import asyncio
import os
import sys
import threading
from pathlib import Path
from typing import Literal

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the pipeline's graph.py lives at the repo root

import anthropic
from anthropic.lib.streaming import MessageStream
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel, Field

import graph as g

JUDGE_MODEL = "claude-opus-5"  # deliberately not the pipeline's model, so the judge isn't grading its own work
PRICES = {"claude-opus-5": (5.00, 25.00), g.MODEL: (2.00, 10.00)}  # $ per million input / output tokens
RECHECK_RESULTS = 5
RECHECK_SNIPPET_CHARS = 1500

client = anthropic.Anthropic()
_usage_lock = threading.Lock()
usage = {}  # model -> [input_tokens, output_tokens]
# ponytail: per-process cap, checked after each call, so it can overshoot by one call (~$0.20 at most).
BUDGET_USD = float(os.getenv("EVAL_BUDGET_USD", "inf"))


class BudgetExceeded(Exception):
    pass


def record_usage(model, response_usage):
    with _usage_lock:
        totals = usage.setdefault(model, [0, 0])
        totals[0] += response_usage.input_tokens
        totals[1] += response_usage.output_tokens
    if cost() >= BUDGET_USD:
        raise BudgetExceeded(f"metered spend ${cost():.2f} reached the ${BUDGET_USD:.2f} cap; stopping")


def cost(model=None):
    models = [model] if model else list(usage)
    return sum(usage.get(m, [0, 0])[0] / 1e6 * PRICES[m][0] + usage.get(m, [0, 0])[1] / 1e6 * PRICES[m][1]
               for m in models)


class JudgeRefused(Exception):
    pass


def ask_judge(schema, prompt):
    """Structured judge call. Refusal fallbacks are on; if the whole chain still refuses, raise."""
    response = client.beta.messages.parse(
        model=JUDGE_MODEL, max_tokens=16000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_format=schema, messages=[{"role": "user", "content": prompt}])
    record_usage(JUDGE_MODEL, response.usage)
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise JudgeRefused(f"judge returned no verdict (stop_reason={response.stop_reason})")
    return response.parsed_output


def meter_pipeline():
    """Count the pipeline's own Claude usage so each run's cost is recorded."""
    original = MessageStream.get_final_message  # the pipeline streams; the judge doesn't, so only it is counted

    def counted(self):
        response = original(self)
        record_usage(g.MODEL, response.usage)
        return response
    MessageStream.get_final_message = counted


class Recheck(BaseModel):
    verdict: Literal["supported", "contradicted", "unconfirmable"]
    reasoning: str = Field(description="one or two sentences citing what the evidence does or doesn't say")
    evidence_urls: list[str] = Field(description="URLs from the evidence that the verdict rests on")


def fresh_evidence(claim, company):
    """Evidence gathered independently of the pipeline: its own query (the claim itself), more and longer results."""
    query = f"{company}: {claim}"[:390]
    results = g.tavily.search(query, max_results=RECHECK_RESULTS)["results"]
    return [{"url": r["url"], "content": r["content"][:RECHECK_SNIPPET_CHARS]} for r in results]


def recheck(claim, company, evidence=None):
    evidence = evidence if evidence is not None else fresh_evidence(claim, company)
    verdict = ask_judge(Recheck, f"""You are fact-checking a claim about {company} for an interview-prep briefing.

Claim: {claim}

Evidence (web search results gathered independently of whoever wrote the claim):
{evidence}

Judge the claim ONLY on this evidence, not on your own knowledge.
- supported: the evidence confirms every specific the claim states (names, titles, dates, figures, attributions).
- contradicted: the evidence states something incompatible with the claim.
- unconfirmable: the evidence neither confirms every specific nor contradicts the claim.""")
    return verdict, evidence


def posting_check(claim, posting):
    """For from_posting claims: is the claim actually stated in the pasted job posting?"""
    return ask_judge(Recheck, f"""Is this claim stated in the job posting below? Judge ONLY against the posting text.
- supported: the posting states every specific in the claim.
- contradicted: the posting says something incompatible with the claim.
- unconfirmable: the posting doesn't state it (fully or at all).
Leave evidence_urls empty.

Claim: {claim}

Job posting:
{posting}""")


class DeepEvalJudge(DeepEvalBaseLLM):
    """Lets DeepEval's metrics use the same Opus judge (Opus 5 rejects the temperature DeepEval's built-in sends)."""

    def load_model(self):
        return client

    def get_model_name(self):
        return JUDGE_MODEL

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is not None:
            return ask_judge(schema, prompt)
        response = client.beta.messages.create(
            model=JUDGE_MODEL, max_tokens=16000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": prompt}])
        record_usage(JUDGE_MODEL, response.usage)
        if response.stop_reason == "refusal":
            raise JudgeRefused("judge refused")
        return next(b.text for b in response.content if b.type == "text")

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        return await asyncio.to_thread(self.generate, prompt, schema)
