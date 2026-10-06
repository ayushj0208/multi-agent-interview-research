import base64
import json
import os
import re
import time
from pathlib import Path

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
# The API's shared secret (see api.py require_token), sent on every call; unset in local dev.
AUTH = {"X-API-Token": os.getenv("API_TOKEN", "")}


def api_headers():
    """The UI's token, plus the visitor's address for the API's per-visitor daily limit (X-Forwarded-For when behind
    Render's proxy, where the first address is the visitor's)."""
    ip = st.context.ip_address  # None when unknown; only ever forward a plain string
    visitor = st.context.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (ip if isinstance(ip, str) else "")
    return {**AUTH, "X-Visitor": visitor[:200]}
# Mirrors api.py's request limits, so visitors see a character counter instead of a rejected request.
MAX_COMPANY_CHARS, MAX_POSTING_CHARS, MAX_FEEDBACK_CHARS = 200, 20_000, 2_000
TYPICAL_REVIEW_S = 60  # the Editor's single review call; measured 55-61s on live runs
STATUS_BADGES = {"verified": ("green", "Verified"), "from_posting": ("blue", "From posting"),
                 "unverified": ("orange", "Unverified"), "contradicted": ("red", "Contradicted"),
                 "off_target": ("gray", "Off-target")}
CATEGORY_TITLES = {"news": "Recent news", "funding": "Funding & business", "tech_stack": "Tech stack",
                   "culture": "Culture", "role_context": "Role context"}
BOX_LABELS = {"researcher": "Researcher", "editor": "Editor", "review": "Your review", "writer": "Writer"}
# One typed line per sentence: the full text (121 characters) is wider than the page column on one line.
DEMO_MODE_LINES = ("Running on a limited free API budget, so this demo uses fewer research passes to stay free.",
                   "More budget means more depth.")
CHECK_VERDICTS = {"yes_with_evidence": ("green", "Yes, with evidence"), "no_history_found": ("gray", "No history found"),
                  "couldnt_determine": ("orange", "Couldn't determine")}

st.set_page_config(page_title="Interview Briefing", page_icon=str(Path(__file__).parent / "assets" / "logo.png"),
                   layout="centered")


def dark_mode():
    # Streamlit has no per-visitor theme API, so dark mode is our own CSS layer. The choice lives in the URL
    # (?theme=dark): it survives refreshes and screen changes, though not a brand-new visit.
    return st.query_params.get("theme") == "dark"


def toggle_theme():
    if st.session_state.dark_toggle:
        st.query_params["theme"] = "dark"
    else:
        st.query_params.pop("theme", None)


# Apple-inspired styling and motion (DESIGN.md); colors, font and radii live in .streamlit/config.toml. The dark
# layer comes after the light one in the same <style>, so it wins wherever it overrides a rule.
here = Path(__file__).parent
css = (here / "style.css").read_text(encoding="utf-8")
if dark_mode():
    css += (here / "dark.css").read_text(encoding="utf-8")
st.html(f"<style>{css}</style>")


def splash_due():
    """The intro plays once per browser session (a refresh or new visit starts a new one), never on reruns, and never
    when the URL resumes a run, so refreshing mid-briefing goes straight back to it."""
    if "splashed" in st.session_state:
        return False
    st.session_state.splashed = True
    return not st.query_params.get("run")


# Pure CSS (style.css .splash), placed right after the stylesheet so nothing unstyled shows first. It never takes
# clicks and the app renders underneath it. Later runs put an empty element in the same slot, so the elements after
# it keep their positions and nothing remounts.
SPLASH = """<div class="splash" aria-hidden="true"><div class="splash-mark">
<svg class="splash-tile" viewBox="0 0 512 512"><defs><linearGradient id="splash-accent" x1="96" y1="96" x2="416"
y2="416" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#2997ff"/><stop offset="1" stop-color="#a78bfa"/>
</linearGradient></defs><rect width="512" height="512" rx="112"/></svg>
<svg class="splash-arc arc-1" viewBox="-6 -6 512 512"><path d="M160 113.15 A128 128 0 0 1 334.85 160"/></svg>
<svg class="splash-arc arc-2" viewBox="-6 -6 512 512"><path d="M352 224 A128 128 0 0 1 224 352"/>
<path class="handle" d="M330 330 L398 398"/></svg>
<svg class="splash-arc arc-3" viewBox="-6 -6 512 512"><path d="M160 334.85 A128 128 0 0 1 113.15 160"/></svg>
<div class="splash-glow"></div>
<svg class="splash-check" viewBox="-6 -6 512 512"><path d="M168 228 L208 268 L284 188"/></svg>
</div><div class="splash-title">Interview briefing</div></div>"""
# st.markdown, not st.html: st.html's sanitizer strips inline SVG.
st.markdown(SPLASH if splash_due() else '<div class="splash-gone"></div>', unsafe_allow_html=True)
# The animated 3D background: plain markup that style.css positions behind the page and animates. A separate call,
# since the HTML sanitizer can drop a leading <style> that shares a call with other markup.
st.html('<div class="bg3d" aria-hidden="true"><div class="orb orb-a"></div><div class="orb orb-b"></div>'
        '<div class="orb orb-c"></div><div class="floor"></div><div class="ring ring-a"></div>'
        '<div class="ring ring-b"></div><div class="ring ring-c"></div></div>')
with st.container(key="theme-toggle"):  # style.css pins it to the top-right corner on every screen
    st.toggle("Dark mode", value=dark_mode(), key="dark_toggle", on_change=toggle_theme)
# Rendered here so it's on every screen and every rerun (screens that stream or rerun never reach the end of the
# script); style.css moves it to the bottom of the page.
st.markdown('<p class="site-footer">&copy; 2026 Ayush Jaiswal. All rights reserved.</p>', unsafe_allow_html=True)


def md(text):
    """Escape $ so Streamlit markdown shows dollar amounts instead of rendering $...$ as LaTeX math."""
    return str(text).replace("$", r"\$")


def go(screen, **values):
    st.session_state.update(screen=screen, **values)
    st.rerun()


def start_over():  # used as a button callback, where Streamlit reruns on its own afterwards
    theme = st.query_params.get("theme")
    st.query_params.clear()
    st.session_state.clear()
    st.session_state.splashed = True  # same visit: starting over doesn't replay the intro
    if theme:  # starting over forgets the run, not the visitor's light/dark choice
        st.query_params["theme"] = theme


def pipeline_settings():
    """The server's pipeline mode (GET /pipeline), fetched once per session; None if the server can't say."""
    if "pipeline" not in st.session_state:
        try:
            response = requests.get(f"{API}/pipeline", headers=api_headers(), timeout=10)
        except requests.RequestException:
            return None  # not cached: the next screen asks again
        if response.status_code != 200:
            return None
        st.session_state.pipeline = response.json()
    return st.session_state.pipeline


def expected_minutes(stage):
    """ "about N minutes" for the mode this server's pipeline runs in (low-cost runs are shorter), or None when the
    server can't say, so the screens fall back to a numberless phrase rather than a number that may be wrong."""
    if (settings := pipeline_settings()) is None:
        return None
    n = settings["expected_minutes"][stage]
    return f"about {n} minute{'' if n == 1 else 's'}"


def sse_events(response):
    event = None
    for line in response.iter_lines(chunk_size=None, decode_unicode=True):
        if line.startswith("event: "):
            event = line.removeprefix("event: ")
        elif line.startswith("data: "):
            yield event, json.loads(line.removeprefix("data: "))


def server_unreachable():
    """The one message for every entry point that can't reach the API, always with a way out."""
    st.error("Can't reach the briefing server right now. Try again in a moment, or start over.")
    st.button("Start over", on_click=start_over, key="unreachable_start_over")
    st.stop()


def fetch_run(run_id):
    try:
        response = requests.get(f"{API}/briefings/{run_id}", headers=api_headers(), timeout=10)
    except requests.RequestException:
        # Every refresh and poll lands here; without a way out, the run ID in the URL would reproduce the crash.
        server_unreachable()
    if response.status_code == 404:
        return None
    if response.status_code != 200:  # e.g. a token mismatch: same way out as an unreachable server
        server_unreachable()
    return response.json()


def restore(run_id):
    """A page refresh loses session state; the run ID in the URL brings the user back to their run."""
    run = fetch_run(run_id)
    if run is None:
        st.query_params.pop("run", None)
        st.session_state.screen = "check"
        return
    st.session_state.update(run_id=run_id, company=run["company_name"])
    if run["state"] == "awaiting_approval":
        st.session_state.update(screen="approval", approval=run["approval"])
    elif run["state"] == "completed":
        st.session_state.update(screen="report", report=run["final_report"])
    else:
        st.session_state.screen = "watching"


# --- Screens -------------------------------------------------------------------

def check_screen():
    """The cheap first step: visa sponsorship history from the company name, before any paid pipeline run."""
    with st.container(key="hero"):
        # The mark as an inline SVG data URI: sharp at any pixel density, no static file serving needed.
        logo = base64.b64encode((here / "assets" / "logo.svg").read_bytes()).decode()
        st.markdown(f'<img class="hero-mark" src="data:image/svg+xml;base64,{logo}" alt="">', unsafe_allow_html=True)
        st.title("Interview briefing")
        st.caption("Enter a company to see whether it has sponsored H-1B work visas before, based on public records. "
                   "It takes about 15 seconds; the full interview briefing comes after, only if you want it.")
        if (pipeline_settings() or {}).get("low_cost_mode"):  # the public demo only; local full-depth dev skips it
            # Typed once by CSS (style.css .typewriter), one line after another. No JS: st.markdown strips scripts.
            # Each line gets its length (--n), the characters typed before it (--prev) and its position (--i).
            lines, typed = [], 0
            for i, text in enumerate(DEMO_MODE_LINES):
                lines.append(f'<span style="--n: {len(text)}; --prev: {typed}; --i: {i}"><span>{text}</span></span>')
                typed += len(text)
            st.markdown(f'<p class="typewriter">{"".join(lines)}</p>', unsafe_allow_html=True)
    with st.form("quick_check"):
        # Keyed so start_over's session_state.clear() also empties it; unkeyed, the typed name would survive.
        company = st.text_input("Company name", key="check_company", max_chars=MAX_COMPANY_CHARS)
        submitted = st.form_submit_button("Quick Check", type="primary")
    if submitted:
        if not company.strip():
            st.error("Enter a company name.")
            return
        try:
            with st.spinner("Checking visa sponsorship records..."):
                response = requests.post(f"{API}/quick-check", json={"company_name": company.strip()},
                                         headers=api_headers(),
                                         timeout=120)
        except requests.RequestException:
            server_unreachable()
        if response.status_code != 200:
            st.error(md(response.json().get("detail", response.text)))
            return
        st.session_state.update(company=company.strip(), check=response.json())
    if check := st.session_state.get("check"):
        render_check(check)
        left, right = st.columns(2)
        if left.button("Generate full briefing", type="primary"):
            go("input", company=check["company_name"])
        right.button("Not interested", on_click=start_over)


def render_check(check):
    st.subheader(f"Quick check: {md(check['company_name'])}")
    answer = check["h1b"]
    color, label = CHECK_VERDICTS[answer["verdict"]]
    with st.container(key="check-h1b"):  # styled as a glass card
        st.markdown(f"**H-1B / work-visa sponsorship** :{color}-badge[{label}]")
        st.markdown(md(answer["summary"]))
        for e in answer["evidence"]:
            st.markdown(f"- [{md(e['url'])}]({e['url']}): “{md(e['quote'])}”")
        if answer["verdict"] == "no_history_found":
            st.caption(f"Searched: {', '.join(check['visa_data_sites'])}. No filings found there doesn't prove "
                       f"the company has never sponsored.")
        st.caption("Past filings show the company has sponsored before, not that it will sponsor this role.")
    st.caption("From public search results; not legal advice. Confirm sponsorship with the employer.")


def input_screen():
    if st.button("← Back"):  # to the quick check, results and company name intact
        go("check", check_company=st.session_state.get("company", ""))
    with st.container(key="hero"):
        st.title("Full briefing")
        st.caption("Paste the full job posting and click Generate Briefing. Three AI agents research the company, "
                   "fact-check what they find, and pause for you to review before anything is written. It takes "
                   f"{expected_minutes('total') or 'a few minutes'}.")
    with st.form("new_briefing"):
        company = st.text_input("Company name", value=st.session_state.get("company", ""), max_chars=MAX_COMPANY_CHARS)
        posting = st.text_area("Job posting", height=320, placeholder="Paste the full job posting text",
                               max_chars=MAX_POSTING_CHARS)
        submitted = st.form_submit_button("Generate Briefing", type="primary")
    if submitted:
        if not company.strip() or not posting.strip():
            st.error("Enter both a company name and the job posting.")
        else:
            go("progress", company=company.strip(),
               request=("/briefings", {"company_name": company.strip(), "job_posting": posting}))


def render_reviewing(placeholder, detail, started):
    elapsed = time.monotonic() - started
    with placeholder.container():
        st.markdown(f"**Cross-checking {detail['claims']} claims against {len(detail['checks'])} independent answers.** "
                    f"This is one long model call that scores every claim and the research as a whole, "
                    f"usually about {TYPICAL_REVIEW_S}s.")
        # Holds at 95% so a slower-than-usual call never looks finished before it is.
        st.progress(min(elapsed / TYPICAL_REVIEW_S, 0.95), text=f"{elapsed:.0f}s so far")
        st.markdown("What the independent checks found:")
        for c in detail["checks"]:
            st.markdown(f"- **{md(c['question'])}**  \n  {md(c['answer'][:240])}")


def progress_screen():
    if "request" not in st.session_state:  # a rerun or refresh mid-stream: the run continues server-side
        return watching_screen()
    path, body = st.session_state.pop("request")
    st.title(f"Researching {md(st.session_state.company)}")
    # What the whole run is and where it ends; the live boxes below say what's happening right now.
    to_review = expected_minutes("to_review")
    st.caption("The Writer is turning the findings you approved into your briefing. This takes about a minute."
               if body.get("action") == "approve" else
               "The Researcher is searching the web, then the Editor independently fact-checks the key claims. It "
               "will pause for your review" + (f", {to_review} from now." if to_review else " in a few minutes."))
    clock = st.empty()
    boxes, reviewing = {}, None

    def box(node):
        if node not in boxes:
            boxes[node] = st.status(BOX_LABELS[node], expanded=True, state="running")
        return boxes[node]

    try:
        with requests.post(f"{API}{path}", json=body, headers=api_headers(), stream=True, timeout=(10, 30)) as response:
            if response.status_code != 200:
                st.error(md(response.json().get("detail", response.text)))
                st.button("Start over", on_click=start_over)
                return
            run_id = response.headers["X-Run-Id"]
            st.session_state.run_id = run_id
            st.query_params["run"] = run_id
            final = None
            for event, d in sse_events(response):
                clock.caption(f"{d['elapsed_s']:.0f}s elapsed")
                if event == "progress":
                    b = box(d["node"])
                    # expanded=True each time: without it the running box collapsed, hiding the review progress bar.
                    b.update(label=f"{BOX_LABELS[d['node']]}: {d['stage']}", state="running", expanded=True)
                    if d["stage"] == "reviewing":
                        reviewing = (b.empty(), d["detail"], time.monotonic())
                        render_reviewing(*reviewing)
                    else:
                        b.write(md(d["message"]))
                        for q in d["detail"].get("queries", []):
                            b.caption(f"{q['category']}: {md(q['query'])}")
                elif event == "heartbeat" and reviewing:
                    render_reviewing(*reviewing)
                elif event == "node_completed":
                    node_completed(box(d["node"]), d)
                    if d["node"] == "editor" and reviewing:
                        reviewing[0].empty()
                        reviewing = None
                elif event in ("awaiting_approval", "completed", "error"):
                    final = (event, d)
    except requests.RequestException as e:
        if "run_id" not in st.session_state:  # the request never reached the server, so no run exists to watch
            server_unreachable()
        st.warning(f"Lost the live connection ({e.__class__.__name__}). The run keeps going on the server.")
        return watching_screen()

    event, d = final or ("lost", {})
    if event == "awaiting_approval":
        go("approval", approval={k: v for k, v in d.items() if k not in ("run_id", "elapsed_s")})
    elif event == "completed":
        go("report", report=d["final_report"])
    elif event == "error":
        if d.get("budget_exhausted"):  # not a crash: the demo's spend cap. Say that, with no stage or raw API text
            st.warning(md(d["message"]))
        else:
            st.error(f"The run failed in the {d.get('node') or 'pipeline'}: {md(d['message'])}")
        st.button("Start over", on_click=start_over)
    else:
        watching_screen()


def node_completed(b, d):
    node = d["node"]
    if node == "researcher":
        thin = ", ".join(d["thin_categories"]) or "none"
        b.write(f"Pass {d['iteration']}: {d['new_findings']} new claims ({d['total_findings']} total). Thin: {thin}")
        b.update(label=f"Researcher: pass {d['iteration']} done", state="complete", expanded=False)
    elif node == "editor":
        b.write(f"Quality {d['quality_score']}/10. " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in d["counts"].items()))
        if d["decision"] == "send_back":
            b.warning(f"Sent back to the Researcher: {md(d['feedback'])}")
        b.update(label=f"Editor: {d['quality_score']}/10, {'sent back' if d['decision'] == 'send_back' else 'passed'}",
                 state="complete", expanded=d["decision"] == "send_back")
    elif node == "review":
        b.write("You approved the findings." if d["action"] == "approve" else f"You sent it back: {md(d['feedback'])}")
        b.update(label="Your review", state="complete", expanded=False)
    elif node == "writer":
        b.update(label=f"Writer: {d['sections']} sections drafted", state="complete", expanded=False)


def watching_screen():
    """No live stream (after a refresh or a dropped connection): poll the saved run until it changes state."""
    run = fetch_run(st.session_state.run_id) if "run_id" in st.session_state else None
    if run is None:
        start_over()
        st.rerun()
    if run["state"] != "running":
        restore(run["run_id"])
        if st.session_state.screen in ("approval", "report"):
            st.rerun()
        st.error(f"This run stopped before finishing. Last status: {md(run['status'])}")
        render_trace(run["trace"])
        st.button("Start over", on_click=start_over)
        return
    st.title(f"Researching {md(st.session_state.company)}")
    st.info("This run is still going on the server. Live stage details aren't available after a refresh, "
            "so this page checks the saved run every few seconds.")
    st.write(f"Latest step: {md(run['status'])}")
    st.button("Leave (the run keeps going)", on_click=start_over)
    st.caption("Leaving doesn't stop or refund the run. To come back to it later, copy this page's address first.")
    time.sleep(3)
    st.rerun()


def approval_screen():
    a = st.session_state.approval
    st.title(f"Review the findings on {md(st.session_state.company)}")
    st.caption("Here's what the research found, with each claim marked by how well it held up in an independent "
               "check. Approve to have your briefing written from these, or send it back with a note if something "
               "important is missing or wrong.")
    col1, col2 = st.columns([1, 3])
    col1.metric("Editor's quality score", f"{a['quality_score']}/10")
    col2.markdown(" ".join(f":{STATUS_BADGES[s][0]}-badge[{STATUS_BADGES[s][1]}: {n}]" for s, n in a["counts"].items()))
    if a.get("passes_exhausted"):
        st.warning(f"**Reached review at a lower score because research passes ran out.** Below "
                   f"{a['pass_threshold']}/10 the Editor normally sends research back automatically, but this run "
                   f"has used all its automatic research passes. Approve it as it is, or send it back with specific "
                   f"guidance.")
    st.info(f"**Editor's notes:** {md(a['feedback'])}")
    st.caption("Claim numbers (#) match the numbers the Editor's notes refer to.")

    for category, claims in a["verified_findings"].items():
        with st.expander(f"{CATEGORY_TITLES.get(category, category)} ({len(claims)})", expanded=True):
            for c in sorted(claims, key=lambda c: c["claim_index"]):
                color, label = STATUS_BADGES[c["status"]]
                source = (f" ([source]({c['source_url']}))" if c["source_url"]
                          else " *(from the job posting you provided)*" if c["status"] == "from_posting" else "")
                st.markdown(f"**#{c['claim_index']}** :{color}-badge[{label}] {md(c['claim'])}{source}")
                if c["note"]:
                    st.caption(md(c["note"]))

    resume_path = f"/briefings/{st.session_state.run_id}/resume"
    if st.button("Approve and write the briefing", type="primary"):
        go("progress", request=(resume_path, {"action": "approve", "feedback": ""}))
    left = a["send_backs_left"]
    if left > 0:
        note = st.text_area("Or send it back: what should the Researcher dig into?", max_chars=MAX_FEEDBACK_CHARS)
        if st.button(f"Send back to Researcher ({left} left)"):
            if not note.strip():
                st.error("Say what to fix so the Researcher knows where to look.")
            else:
                go("progress", request=(resume_path, {"action": "send_back", "feedback": note.strip()}))
    else:
        st.caption("This run has used all its send-backs. Approve to finish it.")
    with st.popover("Start over"):  # a second click to confirm: leaving discards research already paid for
        st.markdown("This discards these findings for good. Nothing more is charged: the Writer never runs.")
        st.button("Discard and start over", on_click=start_over)


def report_screen():
    report = st.session_state.report
    slug = re.sub(r"\W+", "-", st.session_state.company.lower()).strip("-")
    download, new = st.columns(2)
    download.download_button("Download .md", report, file_name=f"{slug}-briefing.md", mime="text/markdown")
    new.button("New briefing", on_click=start_over, key="new_briefing_top")  # no scrolling past a long report
    st.caption("Your interview briefing. Download it as a Markdown file, or open the agent trace at the bottom to "
               "see every search and check behind it.")
    # The most important line on the screen: its own line, bold, right above the report it's about.
    st.markdown("**Anything marked as unconfirmed couldn't be independently verified, so double-check it before "
                "relying on it.**")
    with st.container(key="report"):  # styled as a glass card over the animated background
        st.markdown(md(report))
    with st.expander("Agent trace: how this briefing was built"):
        run = fetch_run(st.session_state.run_id)
        render_trace(run["trace"] if run else [])
    st.button("New briefing", on_click=start_over, key="new_briefing_bottom")


def render_trace(trace):
    for t in trace:
        if t["node"] == "researcher":
            st.markdown(f"**Researcher, pass {t['iteration']}**: {t['new_findings']} new claims, "
                        f"{t['total_findings']} total. Thin afterwards: {', '.join(t['thin_categories']) or 'none'}")
            for q in t["queries"]:
                st.caption(f"{q['category']}: {md(q['query'])}")
        elif t["node"] == "editor":
            st.markdown(f"**Editor**: quality {t['quality_score']}/10, "
                        f"{'sent back' if t['decision'] == 'send_back' else 'passed'}. {md(t['feedback'])}")
            # st.table, not st.dataframe: a dataframe is drawn on a canvas that dark-mode CSS can't recolor.
            st.table([{"question": c["question"], "independent answer": c["answer"], "source": c["source_url"]}
                      for c in t["checks"]])
        elif t["node"] == "review":
            st.markdown(f"**Your review**: {t['action'].replace('_', ' ')}" + (f": {md(t['feedback'])}" if t["feedback"] else ""))
        elif t["node"] == "writer":
            st.markdown(f"**Writer**: {t['sections']} sections")


if "screen" not in st.session_state:
    if run_id := st.query_params.get("run"):
        restore(run_id)
    else:
        st.session_state.screen = "check"

{"check": check_screen, "input": input_screen, "progress": progress_screen, "watching": watching_screen,
 "approval": approval_screen, "report": report_screen}[st.session_state.screen]()
