"""Drives the Streamlit app end to end against the real API running in-process with a fake LLM. No API spend."""
import contextlib
import os
import threading
import time

import streamlit as st
import uvicorn
from langgraph.types import Command
from streamlit.testing.v1 import AppTest

import test_api  # dummy keys + fakes
import api
import graph as g

PORT = 8765
os.environ["API_URL"] = f"http://127.0.0.1:{PORT}"


def serve():
    server = uvicorn.Server(uvicorn.Config(api.app, port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)


def texts(at):
    return " ".join(str(e.value) for kind in ("markdown", "caption", "title", "info", "warning", "error")
                    for e in getattr(at, kind))


def assert_no_raw_dollars(at):
    """Every $ that reaches Streamlit markdown must be escaped, or $...$ spans render as LaTeX math."""
    rendered = [str(e.value) for kind in ("markdown", "caption", "title", "info", "warning", "error")
                for e in getattr(at, kind)]
    with_dollars = [t for t in rendered if "$" in t]
    assert with_dollars, "fixture should put dollar amounts on this screen"
    for text in with_dollars:
        assert "$" not in text.replace("\\$", ""), f"unescaped $ in: {text[:120]}"


def open_app(run_id=None):
    at = AppTest.from_file("app.py", default_timeout=30)
    if run_id:
        at.query_params["run"] = run_id
    return at.run()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def quick_checked(company="Sony Interactive Entertainment"):
    at = open_app()
    at.text_input[0].input(company)
    return button(at, "Quick Check").click().run()


def test_full_flow_through_the_ui():
    prompts = test_api.setup_fakes()
    at = quick_checked()  # the cheap first step, with sources
    assert "Yes, with evidence" in texts(at) and "Couldn't determine" in texts(at)
    assert "whether it has sponsored H-1B work visas before" in texts(at)  # orientation line
    assert test_api.test_routing.LCA_URL in texts(at) and "doesn't mean no" in texts(at)
    assert g.QueryPlan not in prompts, "the quick check must not start the paid pipeline"

    button(at, "Generate full briefing").click().run()
    assert at.text_input[0].value == "Sony Interactive Entertainment"  # carried over from the quick check
    at.text_area[0].input(test_api.test_routing.POSTING)
    button(at, "Generate Briefing").click().run()  # streams until the approval pause

    assert "Review the findings on Sony Interactive Entertainment" in texts(at)
    assert "Approve to have your briefing written from these" in texts(at)  # what they're deciding
    assert "Off-target" in texts(at) and "Sony New Media Solutions" in texts(at)
    assert_no_raw_dollars(at)  # the Editor's notes carry "$182K-289K" and "$31.09 billion ... $3.07 billion"
    for n in (0, 1, 2):  # claim numbers the Editor's notes can refer to
        assert f"**#{n}**" in texts(at)
    assert "research passes ran out" not in texts(at)  # 8/10 passed on its own merits
    run_id = at.session_state.run_id
    assert at.query_params["run"] == [run_id] or at.query_params["run"] == run_id

    refreshed = open_app(run_id)  # a browser refresh while paused must land back on the approval screen
    assert "Review the findings" in texts(refreshed)

    next(b for b in at.button if b.label.startswith("Approve")).click().run()
    assert "# Sony Interactive Entertainment: Interview Briefing" in texts(at)
    # The unconfirmed-claims warning stands alone, in bold, not mid-sentence.
    assert any(str(m.value).startswith("**Anything marked as unconfirmed") for m in at.markdown)
    assert any(b.label == "Download .md" for b in at.get("download_button"))
    at.expander[-1].open = True  # trace view: the Editor entry repeats the dollar-laden feedback
    assert_no_raw_dollars(at)
    assert "Revenue $31.09 billion" in test_api.TestClient(api.app).get(f"/briefings/{run_id}").json()["final_report"]

    refreshed = open_app(run_id)  # and a refresh after completion shows the report
    assert "Interview Briefing" in texts(refreshed)

    assert [b.label for b in at.button].count("New briefing") == 2  # at the top as well as below a long report
    button(at, "New briefing").click().run()  # the top one
    assert any(b.label == "Quick Check" for b in at.button) and "run" not in at.query_params


def test_back_from_posting_keeps_the_quick_check():
    prompts = test_api.setup_fakes()
    at = quick_checked()
    button(at, "Generate full briefing").click().run()
    button(at, "← Back").click().run()
    assert "Yes, with evidence" in texts(at) and at.text_input[0].value == "Sony Interactive Entertainment"
    assert any(b.label == "Generate full briefing" for b in at.button)
    assert g.QueryPlan not in prompts


def test_start_over_from_approval_discards_without_writing():
    prompts = test_api.setup_fakes()
    client = test_api.TestClient(api.app)
    run_id, _ = test_api.start(client)
    at = open_app(run_id)
    button(at, "Discard and start over").click().run()  # the confirm step inside the "Start over" popover
    assert any(b.label == "Quick Check" for b in at.button) and "run" not in at.query_params
    assert g.Briefing not in prompts, "the Writer must never run"
    assert client.get(f"/briefings/{run_id}").json()["state"] == "awaiting_approval"


def test_every_entry_point_fails_the_same_way_when_the_server_is_down():
    os.environ["API_URL"] = "http://127.0.0.1:9"  # nothing listens here
    try:
        at = open_app("a-run-id-from-the-url")  # a refresh mid-run
        assert "Can't reach the briefing server" in texts(at)
        button(at, "Start over").click().run()
        assert any(b.label == "Quick Check" for b in at.button) and "run" not in at.query_params

        at.text_input[0].input("Sony Interactive Entertainment")
        button(at, "Quick Check").click().run()  # the home screen's entry point
        assert "Can't reach the briefing server" in texts(at) and any(b.label == "Start over" for b in at.button)

        at = AppTest.from_file("app.py", default_timeout=30)  # the posting screen's entry point
        at.session_state["screen"], at.session_state["company"] = "input", "Sony Interactive Entertainment"
        at.run()
        at.text_area[0].input(test_api.test_routing.POSTING)
        button(at, "Generate Briefing").click().run()
        assert "Can't reach the briefing server" in texts(at), "must say so, not silently bounce to the start"
        assert not any(b.label == "Quick Check" for b in at.button)
    finally:
        os.environ["API_URL"] = f"http://127.0.0.1:{PORT}"


def test_refreshed_mid_run_visitor_can_leave_and_the_run_finishes():
    gate = threading.Event()
    test_api.setup_fakes(gate)
    client = test_api.TestClient(api.app)
    run_id, _ = test_api.start(client)
    _, thread = api.start_run(run_id, Command(resume={"action": "approve", "feedback": ""}))  # Writer held: running
    real_sleep = time.sleep
    time.sleep = lambda seconds: st.stop()  # end the watching screen's poll after one render instead of looping
    try:
        at = open_app(run_id)
        assert "still going on the server" in texts(at) and "doesn't stop or refund the run" in texts(at)
        button(at, "Leave (the run keeps going)").click().run()
        assert any(b.label == "Quick Check" for b in at.button) and "run" not in at.query_params
    finally:
        time.sleep = real_sleep
        gate.set()
        thread.join(10)
    assert client.get(f"/briefings/{run_id}").json()["state"] == "completed", "leaving must not stop the run"


def test_not_interested_starts_over_without_running_the_pipeline():
    prompts = test_api.setup_fakes()
    at = quick_checked()
    button(at, "Not interested").click().run()
    assert at.text_input[0].value == "" and not at.subheader  # back to an empty first screen, results gone
    assert any(b.label == "Quick Check" for b in at.button)
    assert g.QueryPlan not in prompts


@contextlib.contextmanager
def low_cost_server():
    """The in-process API as LOW_COST_MODE=1 configures it (test_routing checks graph sets exactly these)."""
    saved = g.LOW_COST_MODE, g.MAX_HUMAN_SEND_BACKS, g.EXPECTED_MINUTES
    g.LOW_COST_MODE, g.MAX_HUMAN_SEND_BACKS, g.EXPECTED_MINUTES = True, 1, {"to_review": 2, "total": 3}
    try:
        yield
    finally:
        g.LOW_COST_MODE, g.MAX_HUMAN_SEND_BACKS, g.EXPECTED_MINUTES = saved


def posting_screen():
    at = quick_checked()
    return button(at, "Generate full briefing").click().run()


def test_posting_screen_time_estimate_follows_the_servers_mode():
    test_api.setup_fakes()
    assert "It takes about 4 minutes." in texts(posting_screen())
    with low_cost_server():
        assert "It takes about 3 minutes." in texts(posting_screen())


def test_time_estimate_has_no_number_when_the_server_cant_say():
    os.environ["API_URL"] = "http://127.0.0.1:9"  # nothing listens here
    try:
        at = AppTest.from_file("app.py", default_timeout=30)
        at.session_state["screen"], at.session_state["company"] = "input", "Sony Interactive Entertainment"
        at.run()
        assert "It takes a few minutes." in texts(at)
    finally:
        os.environ["API_URL"] = f"http://127.0.0.1:{PORT}"


def test_progress_line_gives_this_modes_time_to_review():
    for server, expected in ((contextlib.nullcontext, "about 3 minutes from now"),
                             (low_cost_server, "about 2 minutes from now")):
        test_api.setup_fakes()
        fake = g.ask

        def review_fails(model, prompt, attempts=1):  # keeps the progress screen up so its line can be read
            if model is g.EditorReview:
                raise RuntimeError("stopped at the review")
            return fake(model, prompt, attempts)
        g.ask = review_fails
        with server():
            at = posting_screen()
            at.text_area[0].input(test_api.test_routing.POSTING)
            button(at, "Generate Briefing").click().run()
        assert f"It will pause for your review, {expected}." in texts(at) and "The run failed" in texts(at)


def test_low_cost_visitor_gets_one_send_back():
    test_api.setup_fakes()
    with low_cost_server():
        run_id, _ = test_api.start(test_api.TestClient(api.app))
        at = open_app(run_id)
    assert any(b.label == "Send back to Researcher (1 left)" for b in at.button)


def test_ui_sends_the_api_token_and_a_mismatch_fails_cleanly():
    test_api.setup_fakes()
    saved, api.API_TOKEN = api.API_TOKEN, "shared-secret"
    try:
        os.environ["API_TOKEN"] = "shared-secret"  # the UI service's copy
        assert "Yes, with evidence" in texts(quick_checked()), "the UI must send the token on its calls"

        os.environ["API_TOKEN"] = "stale-secret"  # misconfigured deploy: no crash, no results
        at = quick_checked()
        assert not at.exception and "Yes, with evidence" not in texts(at) and at.error
    finally:
        api.API_TOKEN = saved
        os.environ.pop("API_TOKEN", None)


def test_demo_mode_line_shows_only_in_low_cost_mode():
    def typewriter(at):
        return [str(m.value) for m in at.markdown if 'class="typewriter"' in str(m.value)]
    test_api.setup_fakes()
    assert typewriter(open_app()) == [], "full-depth dev shows no demo line"
    with low_cost_server():
        lines = typewriter(open_app())
    first = "Running on a limited free API budget, so this demo uses fewer research passes to stay free."
    second = "More budget means more depth."
    assert len(lines) == 1 and first in lines[0] and second in lines[0]
    # Typed one after the other: the second line starts once the first line's characters are done.
    assert f"--n: {len(first)}; --prev: 0; --i: 0" in lines[0]
    assert f"--n: {len(second)}; --prev: {len(first)}; --i: 1" in lines[0]


def test_used_up_budget_shows_the_demo_message_not_a_raw_error():
    test_api.setup_fakes()
    with test_api.budget_used_up():
        at = quick_checked()  # the home screen
        assert g.BUDGET_MESSAGE in texts(at) and "Error code" not in texts(at)

        at = AppTest.from_file("app.py", default_timeout=30)  # a full briefing run
        at.session_state["screen"], at.session_state["company"] = "input", "Sony Interactive Entertainment"
        at.run()
        at.text_area[0].input(test_api.test_routing.POSTING)
        button(at, "Generate Briefing").click().run()
    assert g.BUDGET_MESSAGE in [str(w.value) for w in at.warning], "shown as a warning, not a crash"
    assert "The run failed" not in texts(at) and "Error code" not in texts(at)
    assert any(b.label == "Start over" for b in at.button)


def dark_css_applied(at):
    return any("color-scheme: dark" in e.proto.body for e in at.get("html"))


def test_dark_mode_toggle_survives_screens_refresh_and_start_over():
    test_api.setup_fakes()
    at = open_app()
    assert at.toggle[0].label == "Dark mode" and not at.toggle[0].value and not dark_css_applied(at)

    at.toggle[0].set_value(True).run()
    assert dark_css_applied(at) and at.query_params["theme"] in ("dark", ["dark"])

    at.text_input[0].input("Sony Interactive Entertainment")
    button(at, "Quick Check").click().run()  # a different screen state: still dark
    assert dark_css_applied(at) and "Yes, with evidence" in texts(at)
    button(at, "Not interested").click().run()  # starting over forgets the run, not the theme
    assert dark_css_applied(at) and at.toggle[0].value and at.query_params["theme"] in ("dark", ["dark"])

    refreshed = AppTest.from_file("app.py", default_timeout=30)
    refreshed.query_params["theme"] = "dark"  # a refresh keeps ?theme=dark in the address
    refreshed.run()
    assert dark_css_applied(refreshed) and refreshed.toggle[0].value

    at.toggle[0].set_value(False).run()
    assert not dark_css_applied(at) and "theme" not in at.query_params


def test_send_back_is_hidden_once_the_cap_is_used():
    test_api.setup_fakes()
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    run_id, _ = test_api.start(client)

    at = open_app(run_id)
    assert any(b.label == "Send back to Researcher (2 left)" for b in at.button)

    for _ in range(2):
        test_api.resume(client, run_id, "send_back", "look at studio headcount")
    at = open_app(run_id)
    labels = [b.label for b in at.button]
    assert not any(label.startswith("Send back") for label in labels) and "Approve and write the briefing" in labels
    assert "used all its send-backs" in texts(at)


def test_low_score_explains_why_it_reached_review():
    test_api.setup_fakes(quality_score=4)  # below the 7/10 cutoff: the Editor keeps sending back until passes run out
    from fastapi.testclient import TestClient
    run_id, events = test_api.start(TestClient(api.app))
    assert [d["decision"] for n, d in events if n == "node_completed" and d["node"] == "editor"][-1] == "pass"

    at = open_app(run_id)
    assert "4/10" in " ".join(str(m.value) for m in at.metric)
    assert "Reached review at a lower score because research passes ran out" in texts(at)


if __name__ == "__main__":
    serve()
    test_full_flow_through_the_ui()
    test_back_from_posting_keeps_the_quick_check()
    test_start_over_from_approval_discards_without_writing()
    test_every_entry_point_fails_the_same_way_when_the_server_is_down()
    test_refreshed_mid_run_visitor_can_leave_and_the_run_finishes()
    test_not_interested_starts_over_without_running_the_pipeline()
    test_dark_mode_toggle_survives_screens_refresh_and_start_over()
    test_posting_screen_time_estimate_follows_the_servers_mode()
    test_time_estimate_has_no_number_when_the_server_cant_say()
    test_progress_line_gives_this_modes_time_to_review()
    test_low_cost_visitor_gets_one_send_back()
    test_used_up_budget_shows_the_demo_message_not_a_raw_error()
    test_demo_mode_line_shows_only_in_low_cost_mode()
    test_ui_sends_the_api_token_and_a_mismatch_fails_cleanly()
    test_send_back_is_hidden_once_the_cap_is_used()
    test_low_score_explains_why_it_reached_review()
    print("ok")
