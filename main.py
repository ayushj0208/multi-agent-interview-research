import argparse
import json
import re
import sys
import time
from pathlib import Path

from graph import initial_state, stream_auto_approved

sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser(description="Generate an interview-prep briefing on a company and role.")
parser.add_argument("company", help='company name, e.g. "Stripe"')
parser.add_argument("job_posting", help="path to a text file containing the job posting")
args = parser.parse_args()

start = time.perf_counter()
state = initial_state(args.company, Path(args.job_posting).read_text(encoding="utf-8"))
out = Path("output")
out.mkdir(exist_ok=True)
slug = re.sub(r"\W+", "-", args.company.lower()).strip("-")
for state in stream_auto_approved(state):  # no human at the CLI: the review pause is auto-approved
    print(f"[{time.perf_counter() - start:6.1f}s] {state['status']}")
    (out / f"{slug}.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

(out / f"{slug}.md").write_text(state["final_report"], encoding="utf-8")

print("\n" + state["final_report"])
print(f"\nDone in {time.perf_counter() - start:.1f}s -> output/{slug}.md, output/{slug}.json")
