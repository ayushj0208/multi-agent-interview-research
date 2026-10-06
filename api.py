import datetime
import hashlib
import hmac
import json
import logging
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import AfterValidator, BaseModel, Field, model_validator

import graph as g
from graph import builder, initial_state
from quick_check import quick_check


# Postgres when deployed (DATABASE_URL set): run checkpoints, plus today's usage counts (see take) so that the free
# tier's sleep-and-restart can't reset the daily caps.
pool = None
if os.getenv("DATABASE_URL"):
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(os.environ["DATABASE_URL"], kwargs={"autocommit": True, "prepare_threshold": 0}, open=True)


def make_checkpointer():
    if pool:
        from langgraph.checkpoint.postgres import PostgresSaver

        saver = PostgresSaver(pool)
        saver.setup()
        with pool.connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS api_usage (day date, key text, n int NOT NULL, PRIMARY KEY (day, key))")
        return saver
    # Local dev only: SQLite's write lock makes it unsuitable once deployed.
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False))


log = logging.getLogger("uvicorn.error")  # shows up in uvicorn's (and Render's) logs

# Shared secret between the Streamlit UI and this API, so only the UI can start paid work. Set the same random
# value on both deployed services; unset (local dev) means no check.
API_TOKEN = os.getenv("API_TOKEN")
if os.getenv("RENDER") and not API_TOKEN:  # Render sets RENDER on every service: never serve paid endpoints unguarded
    raise RuntimeError("API_TOKEN must be set on the deployed API, with the same value as on the UI service")
if g.LOW_COST_MODE and not API_TOKEN:
    log.warning("LOW_COST_MODE is on (public demo) but API_TOKEN is unset: anyone can call the paid endpoints")


def require_token(x_api_token: str = Header(default="")):
    if API_TOKEN and not hmac.compare_digest(x_api_token.encode(), API_TOKEN.encode()):
        raise HTTPException(401, "unauthorized")


# ponytail: in-memory caps, right for one server instance; move to Redis if the API ever runs on several.
MAX_CONCURRENT_RUNS = 2  # pipeline runs in flight at once, resumes included
MAX_CONCURRENT_QUICK_CHECKS = 3
DAILY_LIMITS = {"briefings": int(os.getenv("DAILY_BRIEFING_LIMIT", "20")),
                "quick_checks": int(os.getenv("DAILY_QUICK_CHECK_LIMIT", "100"))}
# Per visitor, so one person can't use up the whole demo for everyone else. The visitor is the client address the UI
# forwards (X-Visitor); it's trusted only because the request already carried the UI's token.
# ponytail: an address can be changed (VPN, mobile network), so this is fairness, not a hard bound; the daily caps
# above and the Anthropic spend limit are the hard bounds.
VISITOR_LIMITS = {"briefings": int(os.getenv("VISITOR_BRIEFING_LIMIT", "3")),
                  "quick_checks": int(os.getenv("VISITOR_QUICK_CHECK_LIMIT", "15"))}
BUSY_MESSAGE = "The demo is busy right now. Please try again in a few minutes."
DAILY_MESSAGE = "The demo has reached its limit for today. Please try again tomorrow."
VISITOR_MESSAGE = "You've reached today's limit for this demo. Please try again tomorrow."
GENERIC_ERROR = "Something went wrong on our side. Please start over and try again."

# Once deployed (token set), no public /docs, /redoc or /openapi.json: they'd hand anyone a map of the API.
NO_DOCS = {"docs_url": None, "redoc_url": None, "openapi_url": None} if API_TOKEN else {}
app = FastAPI(title="Company & Role Research Assistant", dependencies=[Depends(require_token)], **NO_DOCS)
graph = builder.compile(checkpointer=make_checkpointer())
runs: dict[str, threading.Thread] = {}  # in-flight runs, so GET can tell "running" from "stopped"
runs_lock = threading.Lock()
usage = {"day": None, "briefings": 0, "quick_checks": 0}  # today's counts (UTC) when there's no database
quick_checks_running = 0
usage_lock = threading.Lock()
HEARTBEAT_S = 5


def take(key, limit):
    """Counts one use of `key` toward today's `limit` (UTC); False, counting nothing, once it's reached."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    if limit <= 0:
        return False
    if pool:  # one atomic statement: concurrent requests can't both slip past the limit
        with pool.connection() as conn:
            return conn.execute(
                "INSERT INTO api_usage (day, key, n) VALUES (%s, %s, 1) ON CONFLICT (day, key) "
                "DO UPDATE SET n = api_usage.n + 1 WHERE api_usage.n < %s RETURNING n", (today, key, limit)).fetchone() is not None
    with usage_lock:
        if usage["day"] != today:
            usage.clear()
            usage.update(day=today, briefings=0, quick_checks=0)
        if usage.get(key, 0) >= limit:
            return False
        usage[key] = usage.get(key, 0) + 1
        return True


def give_back(key):
    if pool:
        with pool.connection() as conn:
            conn.execute("UPDATE api_usage SET n = n - 1 WHERE day = %s AND key = %s AND n > 0",
                         (datetime.datetime.now(datetime.timezone.utc).date(), key))
        return
    with usage_lock:
        usage[key] = usage.get(key, 0) - 1


def visitor_key(x_visitor):
    """A short hash of the visitor's address: enough to count by, without storing addresses."""
    return hashlib.sha256(x_visitor.encode()).hexdigest()[:16] if x_visitor else None


def take_daily(kind, visitor=None):
    """Count one use toward today's caps (the demo's, then the visitor's), or refuse with 429."""
    if not take(kind, DAILY_LIMITS[kind]):
        raise HTTPException(429, DAILY_MESSAGE)
    if visitor and not take(f"{kind}:{visitor}", VISITOR_LIMITS[kind]):
        give_back(kind)
        raise HTTPException(429, VISITOR_MESSAGE)


def give_back_daily(kind, visitor=None):  # the request was refused for another reason after counting: don't charge it
    give_back(kind)
    if visitor:
        give_back(f"{kind}:{visitor}")

SUMMARY_FIELDS = {"researcher": ("new_findings", "total_findings", "thin_categories"),
                  "editor": ("quality_score", "counts", "decision", "feedback"),
                  "review": ("action", "feedback"),
                  "writer": ("sections",)}


# Every one of these lands in several paid prompts: cap them so one request can't cost dollars instead of cents.
# The longest real posting in the evals is about 6,400 characters.
MAX_COMPANY_CHARS, MAX_POSTING_CHARS, MAX_FEEDBACK_CHARS = 200, 20_000, 2_000


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # keeps tabs and newlines; a NUL would also break Postgres


def clean_text(text):
    return CONTROL_CHARS.sub("", text)


def one_line(name):
    """A company name is one line: it's quoted into search queries and prompts, where line breaks could pose as
    instructions."""
    name = " ".join(clean_text(name).split())
    if not name:
        raise ValueError("company name can't be blank")
    return name


CompanyName = Annotated[str, Field(min_length=1, max_length=MAX_COMPANY_CHARS), AfterValidator(one_line)]


class BriefingRequest(BaseModel):
    company_name: CompanyName
    job_posting: Annotated[str, Field(min_length=1, max_length=MAX_POSTING_CHARS), AfterValidator(clean_text)]


class QuickCheckRequest(BaseModel):
    company_name: CompanyName


class ResumeRequest(BaseModel):
    action: Literal["approve", "send_back"]
    feedback: Annotated[str, Field(max_length=MAX_FEEDBACK_CHARS), AfterValidator(clean_text)] = ""

    @model_validator(mode="after")
    def send_back_needs_feedback(self):
        if self.action == "send_back" and not self.feedback.strip():
            raise ValueError("feedback is required when sending research back")
        return self


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def run_graph(run_id, graph_input, events):
    config = {"configurable": {"thread_id": run_id}}
    start = time.perf_counter()

    def emit(event, **data):
        events.put(sse(event, {"run_id": run_id, "elapsed_s": round(time.perf_counter() - start, 1), **data}))

    if isinstance(graph_input, Command):
        emit("run_resumed", action=graph_input.resume["action"])
    else:
        emit("run_started", company_name=graph_input["company_name"])
    node = None
    try:
        for mode, chunk in graph.stream(graph_input, config, stream_mode=["custom", "updates"]):
            if mode == "custom":
                node = chunk["node"]
                emit("progress", **chunk)
                continue
            for node, update in chunk.items():
                if node == "__interrupt__":  # reported below from the saved snapshot
                    continue
                entry = update["trace"][-1]
                emit("node_completed", node=node, iteration=entry["iteration"], status=update["status"],
                     **{k: entry[k] for k in SUMMARY_FIELDS[node]})
        snapshot = graph.get_state(config)
        if snapshot.interrupts:
            emit("awaiting_approval", **snapshot.interrupts[0].value)
        else:
            emit("completed", final_report=snapshot.values["final_report"])
    except Exception as e:  # the checkpoint keeps everything up to the failing node
        budget = isinstance(e, g.BudgetExhausted)
        if not budget:  # the real error stays in the server log: it can name the database host or quote API responses
            log.exception("run %s failed in %s", run_id, node or "pipeline")
        emit("error", node=node, message=str(e) if budget else GENERIC_ERROR, budget_exhausted=budget)
    finally:
        with runs_lock:
            runs.pop(run_id, None)
        events.put(None)


def start_run(run_id, graph_input):
    events = queue.Queue()
    # The run lives on its own thread, not the request: a client disconnect only ends event_stream(),
    # never the (paid) run, which finishes and checkpoints regardless.
    thread = threading.Thread(target=run_graph, args=(run_id, graph_input, events))
    with runs_lock:
        if run_id in runs:
            raise HTTPException(409, "this run is already in progress")
        if len(runs) >= MAX_CONCURRENT_RUNS:
            raise HTTPException(429, BUSY_MESSAGE)
        runs[run_id] = thread
    thread.start()
    return events, thread


def event_stream(run_id, events):
    start = time.perf_counter()
    while True:
        try:
            item = events.get(timeout=HEARTBEAT_S)
        except queue.Empty:  # keeps long silent stages (the Editor's review call) visibly alive
            yield sse("heartbeat", {"run_id": run_id, "elapsed_s": round(time.perf_counter() - start, 1)})
            continue
        if item is None:
            return
        yield item


def stream_response(run_id, events):
    return StreamingResponse(event_stream(run_id, events), media_type="text/event-stream",
                             headers={"X-Run-Id": run_id, "Cache-Control": "no-cache"})


def snapshot_or_404(run_id):
    snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
    if not snapshot.values:
        raise HTTPException(404, "unknown run_id")
    return snapshot


@app.get("/pipeline")
def pipeline_settings():
    """How deep this server's runs go, so the UI's time estimates match the mode the pipeline is actually in."""
    return {"low_cost_mode": g.LOW_COST_MODE, "research_passes": g.MAX_RETRIES,
            "verification_questions": g.MAX_VERIFICATION_QUESTIONS, "human_send_backs": g.MAX_HUMAN_SEND_BACKS,
            "expected_minutes": g.EXPECTED_MINUTES}


@app.post("/quick-check")
def run_quick_check(req: QuickCheckRequest, x_visitor: str = Header(default="", max_length=200)):
    """The cheap first step (about $0.02): visa sponsorship history, from the company name alone."""
    global quick_checks_running
    with usage_lock:
        if quick_checks_running >= MAX_CONCURRENT_QUICK_CHECKS:
            raise HTTPException(429, BUSY_MESSAGE)
        quick_checks_running += 1
    try:
        take_daily("quick_checks", visitor_key(x_visitor))
        return quick_check(req.company_name.strip())
    except g.BudgetExhausted:
        raise HTTPException(503, g.BUDGET_MESSAGE)
    except HTTPException:
        raise
    except Exception:  # e.g. Claude unavailable with no NIM fallback; the details stay in the server log
        log.exception("quick check failed for %r", req.company_name)
        raise HTTPException(502, "The quick check couldn't finish. Please try again in a moment.")
    finally:
        with usage_lock:
            quick_checks_running -= 1


@app.post("/briefings")
def create_briefing(req: BriefingRequest, x_visitor: str = Header(default="", max_length=200)):
    run_id = uuid.uuid4().hex
    visitor = visitor_key(x_visitor)
    take_daily("briefings", visitor)
    try:
        events, _ = start_run(run_id, initial_state(req.company_name, req.job_posting))
    except HTTPException:  # busy: refused before any work, so it doesn't count toward today's cap
        give_back_daily("briefings", visitor)
        raise
    return stream_response(run_id, events)


@app.post("/briefings/{run_id}/resume")
def resume_briefing(run_id: str, req: ResumeRequest):
    snapshot = snapshot_or_404(run_id)
    if run_id in runs or not snapshot.interrupts:
        raise HTTPException(409, "this run is not waiting for approval")
    if req.action == "send_back" and snapshot.interrupts[0].value["send_backs_left"] <= 0:
        raise HTTPException(409, "no send-backs left for this run; approve to finish it")
    events, _ = start_run(run_id, Command(resume=req.model_dump()))
    return stream_response(run_id, events)


@app.get("/briefings/{run_id}")
def get_briefing(run_id: str):
    snapshot = snapshot_or_404(run_id)
    values = snapshot.values
    if run_id in runs:
        state = "running"
    elif snapshot.interrupts:
        state = "awaiting_approval"
    else:
        state = "completed" if values["final_report"] else "stopped"
    return {"run_id": run_id, "state": state, "status": values["status"], "company_name": values["company_name"],
            "approval": snapshot.interrupts[0].value if snapshot.interrupts else None,
            "final_report": values["final_report"] or None, "trace": values["trace"]}
