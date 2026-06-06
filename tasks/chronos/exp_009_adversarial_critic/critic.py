#!/usr/bin/env python3
"""
Experiment 009 — Adversarial Project Critic (Option B: gemma4:26b local)

Reads project context, applies three adversarial personas, outputs JSON.

Usage:
  python3 critic.py --project ~/REPOS/casasol
  python3 critic.py --project ~/REPOS/casasol --model gemma4:26b
  python3 critic.py --project ~/REPOS/casasol --dry-run   # print context, skip inference
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:26b"
TIMEOUT = 300

RESULTS_DIR = Path(__file__).parent / "results"

# ── Context collection ────────────────────────────────────────────────────────

def git_log(project: Path, n: int = 20) -> str:
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{n}"],
            cwd=project, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or "(no commits)"
    except Exception as e:
        return f"(git log failed: {e})"


def tail_file(path: Path, lines: int = 100) -> str:
    if not path.exists():
        return f"(not found: {path.name})"
    text = path.read_text(encoding="utf-8")
    return "\n".join(text.splitlines()[-lines:])


def head_file(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return f"(not found: {path.name})"
    text = path.read_text(encoding="utf-8")
    return "\n".join(text.splitlines()[:lines])


def collect_context(project: Path) -> dict:
    candidates_brief = ["BRIEF.md", "README.md", "CASASOL.md"]
    candidates_todo  = ["strategy/TODO.md", "tasks/CURRENT_TASK.md", "TODO.md"]

    brief_path = next((project / f for f in candidates_brief if (project / f).exists()), None)
    todo_path  = next((project / f for f in candidates_todo  if (project / f).exists()), None)

    build_log_path = project / "BUILD_LOG.md"

    return {
        "git_log":   git_log(project),
        "build_log": tail_file(build_log_path, lines=100),
        "brief":     head_file(brief_path, lines=80) if brief_path else "(not found)",
        "todo":      head_file(todo_path,  lines=80) if todo_path  else "(not found)",
        "brief_path": str(brief_path) if brief_path else "none",
        "todo_path":  str(todo_path)  if todo_path  else "none",
    }


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an adversarial project critic running three review passes on the same project.
Your job is to find real weaknesses — not surface-level suggestions, not praise.

You will receive project context: recent commits, build log, strategic brief, and open TODO items.

Apply each persona in sequence. Be specific. Quote evidence from the context when you flag something.

PERSONA 1 — SCEPTICAL INVESTOR
Question: Is the moat real or just marketing copy?
Look for: unauditable claims, circular reasoning in the thesis, missing defensibility evidence,
competitor paths that are easier than claimed, market numbers that can't be verified.

PERSONA 2 — DPO / AEPD AUDITOR
Question: Will the compliance architecture survive a real audit?
Look for: gaps in the RoPA / LIA / DPA coverage, missing data subject rights procedures,
vague "lawful basis" claims, unresolved Art. 9 special category exposure,
compliance items that are described but not implemented.

PERSONA 3 — COMPETING ENGINEER
Question: How would I replicate this in a weekend?
Look for: commodity components dressed up as moats, missing technical barriers,
architectural claims that are just configuration choices, open-source alternatives
that undercut the differentiation, corners cut that would fail under real load.

---

Output ONLY valid JSON, no preamble, no markdown fences, no explanation:

{
  "personas": {
    "investor":  {"holds_up": [...], "weak": [...], "missing": [...]},
    "dpo":       {"holds_up": [...], "weak": [...], "missing": [...]},
    "engineer":  {"holds_up": [...], "weak": [...], "missing": [...]}
  },
  "top_actions": ["action 1", "action 2", "action 3"],
  "severity_counts": {"high": N, "medium": N, "low": N}
}

Each item in holds_up/weak/missing is a single sentence. Be concrete. Maximum 4 items per bucket.
"""


def build_prompt(ctx: dict) -> str:
    return f"""PROJECT CONTEXT
===============

## Recent commits (last 20)
{ctx['git_log']}

## Build log (last 100 lines)
{ctx['build_log']}

## Strategic brief / README (first 80 lines from {ctx['brief_path']})
{ctx['brief']}

## Open TODO / current task (first 80 lines from {ctx['todo_path']})
{ctx['todo']}

===============
Now apply the three adversarial personas and output the JSON critique.
"""


# ── Inference ─────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str) -> tuple[str, float, dict]:
    t0 = time.time()
    response = requests.post(OLLAMA_URL, json={
        "model":   model,
        "system":  SYSTEM_PROMPT,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.2, "num_predict": 2048},
    }, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()

    input_tokens  = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)
    eval_ns       = data.get("eval_duration", 0)
    tokens_per_s  = output_tokens / (eval_ns / 1e9) if eval_ns > 0 else 0

    token_stats = {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  input_tokens + output_tokens,
        "tokens_per_s":  round(tokens_per_s, 1),
    }
    return data["response"].strip(), time.time() - t0, token_stats


def parse_output(raw: str) -> dict:
    text = raw.strip()
    # strip markdown fences if the model adds them
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_result(data: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    slug = data["ts"].replace(":", "").replace("-", "").replace("+", "")[:15]
    critic_slug = data["critic"].replace(":", "_").replace(".", "_").replace("-", "_")
    filename = f"{critic_slug}_{slug}.json"
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Adversarial Project Critic — Option B (local)")
    parser.add_argument("--project", "-p", required=True, help="Path to the project root")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Print context, skip inference")
    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()
    if not project.is_dir():
        print(f"[ERROR] Not a directory: {project}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Adversarial Critic (Option B — {args.model}) ===")
    print(f"  Project: {project}")

    print("\n  Collecting context...")
    ctx = collect_context(project)
    prompt = build_prompt(ctx)

    print(f"  Context size: {len(prompt)} chars")

    if args.dry_run:
        print("\n--- CONTEXT (dry run) ---")
        print(prompt[:3000], "...(truncated)" if len(prompt) > 3000 else "")
        return

    print(f"\n  Calling {args.model}...")
    try:
        raw, duration, token_stats = call_ollama(prompt, args.model)
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to Ollama. Is it running?", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"[ERROR] Ollama timed out after {TIMEOUT}s", file=sys.stderr)
        sys.exit(1)

    print(f"  Done in {duration:.1f}s")
    print(f"  Tokens — input: {token_stats['input_tokens']}  "
          f"output: {token_stats['output_tokens']}  "
          f"total: {token_stats['total_tokens']}  "
          f"speed: {token_stats['tokens_per_s']} tok/s")

    try:
        parsed = parse_output(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse failed ({e}) — saving raw output")
        parsed = {"raw": raw, "parse_error": str(e)}

    result = {
        "ts":           datetime.now(timezone.utc).isoformat(),
        "critic":       args.model,
        "project":      str(project),
        "duration_s":   round(duration, 2),
        "prompt_chars": len(prompt),
        "tokens":       token_stats,
        **parsed,
    }

    path = save_result(result)
    print(f"\n  Saved → {path.name}")

    # Pretty-print top findings
    if "personas" in result:
        print()
        for persona, findings in result["personas"].items():
            highs = findings.get("missing", []) + findings.get("weak", [])
            if highs:
                print(f"  [{persona.upper()}]")
                for item in highs[:3]:
                    print(f"    ⚠  {item}")
        if result.get("top_actions"):
            print("\n  Top actions:")
            for a in result["top_actions"]:
                print(f"    →  {a}")

    print()


if __name__ == "__main__":
    main()
