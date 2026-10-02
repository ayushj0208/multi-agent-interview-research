# Multi-Agent Company & Role Research Assistant — Claude Code Build Spec

## How to use this file with Claude Code (VS Code)

1. Save this file at the root of your project repo (e.g. `BUILD_SPEC.md`).
2. Install the recommended skills/plugins in **Section 8** first — they make
   Claude Code noticeably better at this specific build.
3. Open the Claude Code panel in VS Code and start with:
   > "Read BUILD_SPEC.md. Start with Phase 1 only — scaffold the LangGraph
   > pipeline with the three agents and get it running end-to-end from a
   > script before touching any API or UI code. Stop and show me the result
   > before moving to Phase 2."
4. Review each phase's output before telling it to continue to the next —
   don't let it run all 5 phases unattended in one go. Catching drift early
   (wrong schema, wrong agent boundaries) is much cheaper than fixing it
   after the UI is built on top of it.

---

## 1. Project Overview

**What it is:** A multi-agent system — Researcher, Editor, Writer — that
collaborates to produce a structured research briefing on a company and job
role, given a company name and a job posting.

**Why this project:** Demonstrates agent orchestration and tool use, not
just an LLM API wrapper. The architecture below (verification loop,
iterative search, structured output, evals) is what separates a "notebook
toy" from a portfolio piece that reads as production-minded.

**Personal angle:** Built for my own job search — interview prep briefings.
Use this framing in the README and resume bullet.

**Target resume bullet (fill in real numbers once built):**
> "Built a multi-agent research system (LangGraph, Python) with
> researcher/editor/writer agents and an independent verification loop;
> generates structured company briefings from live web search in under
> [X] seconds; deployed with a live demo link."

---

## 2. Core User Flow

1. User enters a company name + pastes/links a job posting.
2. **Researcher Agent** runs targeted web searches (company overview, recent
   news, funding, tech stack, culture, role-specific context) and can
   re-search if results are thin — bounded by a max-iteration counter so it
   can't loop forever.
3. **Editor Agent** independently re-verifies the Researcher's claims
   (Chain-of-Verification — see Section 4), deduplicates, flags
   unverified/contradictory items, and structures findings into fixed
   categories. If quality is below threshold, it routes back to the
   Researcher with specific feedback instead of passing bad data forward.
4. **Writer Agent** turns verified findings into a polished briefing
   (structured JSON → markdown → PDF) with a "talking points" section.
5. UI streams live agent status so the user watches the pipeline work
   instead of staring at a spinner. A human-in-the-loop checkpoint lets the
   user approve or send back the research before the Writer finalizes it.

---

## 3. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** | Directed graph + typed shared state = deterministic control flow, checkpointing, time-travel debugging. Highest resume credibility of the current frameworks (CrewAI is faster to prototype but reads as less rigorous; AutoGen is fragmented post-2026 split). |
| LLM | Claude or GPT-4o via API | Abstract behind an interface so it's swappable — useful for the eval section later (compare models). |
| Web search | **Tavily** | AI-native search built for agent workflows, native LangGraph integration, 1,000 free credits/month, returns clean model-ready text in one call. |
| Deep page extraction | **Firecrawl** (optional, for known company domains — investor relations pages, docs) | Full-page Markdown/JSON extraction beyond what a search snippet gives you. |
| Structured output | **Pydantic + `instructor`** | Forces the Writer's output into a strict schema; `instructor` auto-retries the LLM call if validation fails. |
| PDF generation | **Typst** (not WeasyPrint/wkhtmltopdf — both are effectively obsolete for this by 2026: slow, no modern CSS, no JS) | Compiles in milliseconds via a lightweight binary; Writer outputs JSON → Jinja template → Typst markup → PDF. |
| Backend | Python, FastAPI | Wraps the LangGraph pipeline, streams agent status via SSE. |
| Checkpointer | **Postgres (`PostgresSaver`)** in production, SQLite only for local dev | SQLite's write-lock limits make it unsuitable once deployed. |
| Frontend | Streamlit first (to validate the UX fast), then optionally Next.js for the portfolio-grade version | Don't build the polished frontend until the pipeline is proven. |
| Evaluation | **Ragas** + **DeepEval** | Ragas for tool-call sequencing/argument accuracy; DeepEval's FaithfulnessMetric/HallucinationMetric for catching ungrounded claims. |
| Deployment | Render or Railway (backend + managed Postgres), Vercel (frontend if Next.js) | Both handle long-running agent processes without serverless timeout issues. |

---

## 4. Agent Design

### Researcher Agent
- **Input:** company name, job posting text/URL
- **Tools:** Tavily search, optional Firecrawl page fetch
- **Behavior:** Runs multiple targeted searches across fixed categories
  (news, funding, tech stack, culture, role context). After each search,
  evaluates whether the category is adequately covered; if not, reformulates
  the query and searches again — up to a hard `MAX_RETRIES` cap enforced by
  a router node checking `state.iteration_count`, so it always halts
  deterministically regardless of perceived completeness.
- **Output:** raw findings, each tagged with source URL and category.

### Editor Agent — Chain-of-Verification pattern
- **Input:** raw findings from Researcher
- **Behavior:** This is the piece that most portfolio projects skip, and
  the one worth doing properly:
  1. Generates independent verification questions for the Researcher's key
     claims.
  2. Answers those questions by querying the search tools *again*,
     independently — without conditioning on the original draft, to avoid
     just re-confirming the same bias.
  3. Compares the independent answers against the original claims, scores
     confidence, and flags inconsistencies.
  4. Deduplicates and organizes everything into fixed categories.
  5. If quality is below threshold, routes back to the Researcher node with
     specific, actionable feedback (Self-Refine pattern) rather than
     passing flawed data downstream — this prevents "cascading
     hallucination," where an early error gets treated as ground truth by
     every later step.
- **Output:** structured, verified findings (JSON matching a fixed schema).

### Writer Agent
- **Input:** structured findings from Editor
- **Behavior:** Generates the final briefing in a fixed template — intro,
  categorized findings, 3-5 interview talking points, "what to ask them."
  Output is constrained via a Pydantic schema (using `instructor` for
  auto-retry on malformed output) so required fields are always present.
- **Output:** structured JSON → rendered markdown in the UI → Typst-compiled
  PDF for download.

### Shared State (LangGraph — draft schema)
```python
class ResearchState(TypedDict):
    company_name: str
    job_posting: str
    raw_findings: list[dict]        # researcher output
    verified_findings: dict         # editor output
    final_report: str               # writer output
    iteration_count: int            # researcher retry guard
    editor_feedback: str | None     # set when editor sends work back
    status: str                     # for streaming UI updates
    trace: list[dict]               # full step log — feeds eval + demo UI
```

---

## 5. UI/UX

- **Input screen:** company name + job posting field, one "Generate
  Briefing" button.
- **Progress view:** live-updating agent status list ("Researcher
  searching... Editor verifying 12 claims... sending back to Researcher —
  2 gaps found... Writer drafting..."). This is the single highest-value UX
  element for a demo video — it's what makes the multi-agent architecture
  visible rather than invisible.
- **Human-in-the-loop checkpoint:** before the Writer finalizes, pause
  (LangGraph `interrupt()`) and show the user the verified findings with an
  approve/edit/send-back option. Demonstrates a steerable system, not a
  black box — this is a specific signal the research flagged as valuable.
- **Report view:** rendered markdown + "Download PDF" + collapsible "view
  agent trace" section (great for walking an interviewer through how it
  actually works).
- **History page (stretch):** past briefings.

---

## 6. Implementation Phases

**Phase 1 — Core pipeline (no UI).** Build the LangGraph graph — all three
agents, the verification loop, the retry-guarded research loop — as a
Python script. Validate against 2-3 real companies. This is where most of
the actual engineering happens; don't rush to the UI before this is solid.

**Phase 2 — API layer.** Wrap the graph in FastAPI with an SSE endpoint
streaming agent status. Add the Postgres checkpointer.

**Phase 3 — Frontend.** Streamlit first to validate the flow end-to-end,
then optionally rebuild in Next.js. Add the human-in-the-loop approval
screen.

**Phase 4 — Evaluation.** Build a small `evals/` directory: 5-10 test
companies, scored with Ragas (tool-call sequencing) and DeepEval
(faithfulness/hallucination). Put a results table in the README — this is
what proves you treat the system as engineering, not a demo toy.

**Phase 5 — Polish & deploy.** Deploy to Render/Railway + Vercel, record a
short demo video, write the README (Section 7), fill in the real resume
bullet numbers.

---

## 7. README Checklist

- [ ] One-paragraph problem statement + why multi-agent, not a single call
- [ ] Architecture diagram (agents + data flow + verification loop)
- [ ] Explicit note on what it borrows from known architectures — Stanford
      STORM's outline-first approach, GPT-Researcher's parallel scatter-
      gather, LangChain's Open Deep Research supervisor-worker pattern —
      this signals you know the landscape, not just your own repo
- [ ] Live demo link + short demo GIF/video
- [ ] `evals/` results table (Ragas + DeepEval scores)
- [ ] Known limitations / what you'd improve next
- [ ] Tech stack list

---

## 8. Skills & Plugins to Install in Claude Code First

Install these before starting Phase 1 — they make Claude Code meaningfully
better at this specific build. A caution up front: skills and plugins run
with real privileges once installed (some execute actual scripts, not just
instructions). The first two below are maintained by Anthropic or have a
transparent, well-documented codebase; treat community skill libraries the
same way you'd treat any dependency — skim the SKILL.md before trusting it,
don't install-and-forget.

**1. `feature-dev` — Anthropic's official plugin**
End-to-end feature workflow: explores the codebase, architects the
solution, implements it, then runs code review before you merge. Good fit
for working through the phases above.
- Repo: https://github.com/anthropics/claude-plugins-official
- Install (two commands, run as separate prompts in Claude Code):
  ```
  /plugin marketplace add anthropics/claude-plugins-official
  /plugin install feature-dev@claude-plugins-official
  ```

**2. `ponytail` — keeps Claude Code from over-engineering**
Steers the agent toward the simplest solution that actually works (stdlib
first, no unrequested abstractions) without cutting corners on security or
error handling. Genuinely useful here since multi-agent codebases are easy
to over-abstract.
- Repo: https://github.com/DietrichGebert/ponytail (MIT)
- Install (two separate prompts — the install needs two turns to complete):
  ```
  /plugin marketplace add DietrichGebert/ponytail
  /plugin install ponytail@ponytail
  ```
- Needs `node` on your PATH for its two lifecycle hooks; if it's missing,
  the skill content still works, you just lose the always-on nudge.

**3. `langgraph-patterns` and `mcp-architecture`**
Community skills specifically for LangGraph graph orchestration, state
machines, and human-in-the-loop patterns — directly relevant to Section 4
above.
- Repo: https://github.com/frankxai/claude-skills-library (community, MIT
  — unverified beyond what's in the repo itself; read the SKILL.md files
  before installing)
- Install (clone and copy just the folders you need — check the repo's
  actual folder layout first, since community repos restructure):
  ```
  git clone https://github.com/frankxai/claude-skills-library.git
  mkdir -p ~/.claude/skills
  cp -r claude-skills-library/langgraph-patterns ~/.claude/skills/
  cp -r claude-skills-library/mcp-architecture ~/.claude/skills/
  ```

**4. More options (browse, don't blind-install)**
A curated directory of additional community skills, if you want a frontend
or design-focused skill later for the Next.js rebuild in Phase 3:
- https://github.com/ComposioHQ/awesome-claude-skills

---

## 9. Stretch Goals (only after Phases 1-5 are solid)

- Compare multiple companies side by side
- Email/Slack delivery of the briefing
- Memory across sessions (remember companies researched before)
- Model comparison in the eval report (e.g. GPT-4o vs. Claude on
  faithfulness score) — doubles as a stronger README table
