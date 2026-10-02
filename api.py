import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

import graph as g
from graph import builder, initial_state
from quick_check import quick_check


def make_checkpointer():
    if url := os.getenv("DATABASE_URL"):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        saver = PostgresSaver(ConnectionPool(url, kwargs={"autocommit": True, "prepare_threshold": 0}))
        saver.setup()
        return saver
    # Local dev only (BUILD_SPEC: SQLite's write lock makes it unsuitable once deployed).
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False))


app = FastAPI(title="Company & Role Research Assistant")
graph = builder.compile(checkpointer=make_checkpointer())
runs: dict[str, threading.Thread] = {}  # in-flight runs, so GET can tell "running" from "stopped"
runs_lock = threading.Lock()
HEARTBEAT_S = 5

SUMMARY_FIELDS = {"researcher": ("new_findings", "total_findings", "thin_categories"),
                  "editor": ("quality_score", "counts", "decision", "feedback"),
                  "review": ("action", "feedback"),
                  "writer": ("sections",)}


class BriefingRequest(BaseModel):
    company_name: str = Field(min_length=1)
    job_posting: str = Field(min_length=1)


class QuickCheckRequest(BaseModel):
    company_name: str = Field(min_length=1)


class ResumeRequest(BaseModel):
    action: Literal["approve", "send_back"]
    feedback: str = ""

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
        emit("error", node=node, message=str(e), budget_exhausted=isinstance(e, g.BudgetExhausted))
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
def run_quick_check(req: QuickCheckRequest):
    """The cheap first step (about $0.02): visa sponsorship history and E-Verify, from the company name alone."""
    try:
        return quick_check(req.company_name.strip())
    except g.BudgetExhausted:
        raise HTTPException(503, g.BUDGET_MESSAGE)
    except Exception as e:  # e.g. Claude unavailable with no NIM fallback: say so instead of a bare 500
        raise HTTPException(502, f"The quick check failed: {e}")


@app.post("/briefings")
def create_briefing(req: BriefingRequest):
    run_id = uuid.uuid4().hex
    events, _ = start_run(run_id, initial_state(req.company_name, req.job_posting))
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
