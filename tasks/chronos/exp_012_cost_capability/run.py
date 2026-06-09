#!/usr/bin/env python3
"""
Exp 012 — Cost vs Capability: Where the Curve Breaks
Two tasks, four model tiers, pre-scored rubric in rubric.md.

Usage:
  # Local (Ollama)
  python3 run.py --model gemma4:26b --project ~/REPOS/casasol

  # API (Anthropic)
  python3 run.py --model claude-haiku-4-5-20251001 --project ~/REPOS/casasol --api
  python3 run.py --model claude-sonnet-4-6         --project ~/REPOS/casasol --api
  python3 run.py --model claude-opus-4-8           --project ~/REPOS/casasol --api

  # Dry run (print context, skip inference)
  python3 run.py --model gemma4:26b --project ~/REPOS/casasol --dry-run
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 300
REPS           = 3
IDLE_BETWEEN_REPS = 10  # seconds

RESULTS_DIR = Path(__file__).parent / "results"

# ── Context helpers ───────────────────────────────────────────────────────────

def git_log(project: Path, n: int = 20) -> str:
    try:
        r = subprocess.run(["git", "log", "--oneline", f"-{n}"],
                           cwd=project, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "(no commits)"
    except Exception as e:
        return f"(git log failed: {e})"


def read_head(path: Path, lines: int) -> str:
    if not path.exists():
        return f"(not found: {path})"
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[:lines])


def read_tail(path: Path, lines: int) -> str:
    if not path.exists():
        return f"(not found: {path})"
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[-lines:])


def read_full(path: Path) -> str:
    if not path.exists():
        return f"(not found: {path})"
    return path.read_text(encoding="utf-8")


# ── Context bundles ───────────────────────────────────────────────────────────

def context_task_a(project: Path) -> str:
    """DPO compliance task: policy docs + code snippets."""
    return f"""PROJECT: CasaSol.ai — local-first real estate intelligence platform

## BRIEF.md (first 60 lines)
{read_head(project / 'BRIEF.md', 60)}

## compliance/01-ropa.md (Record of Processing Activities)
{read_full(project / 'compliance' / '01-ropa.md')}

## compliance/07-retention-schedule.md
{read_full(project / 'compliance' / '07-retention-schedule.md')}

## scripts/inference_log.py (first 55 lines)
{read_head(project / 'scripts' / 'inference_log.py', 55)}

## scripts/mcp_server.py (first 55 lines)
{read_head(project / 'scripts' / 'mcp_server.py', 55)}
"""


def context_task_b(project: Path) -> str:
    """Implementation gap task: docs + code + git history."""
    return f"""PROJECT: CasaSol.ai — local-first real estate intelligence platform

## BRIEF.md (first 60 lines)
{read_head(project / 'BRIEF.md', 60)}

## BUILD_LOG.md (last 80 lines)
{read_tail(project / 'BUILD_LOG.md', 80)}

## git log (last 20 commits)
{git_log(project, 20)}

## db.py (full)
{read_full(project / 'db.py')}

## config.py (full)
{read_full(project / 'config.py')}

## scripts/mcp_server.py (first 80 lines)
{read_head(project / 'scripts' / 'mcp_server.py', 80)}
"""


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_A = """\
You are a DPO / AEPD Compliance Auditor reviewing a software project for GDPR gaps.

Your job: identify gaps between documented privacy policy, compliance documents, and actual implementation.
Be specific. Quote the document or code file when you cite evidence.
Do NOT praise. Only report gaps and risks.

Focus on:
- Discrepancies between what policy documents say and what the code actually does
- Missing required GDPR documents or procedures
- Data retention: what policy says vs what code implements
- Access control on endpoints that handle personal data
- Audit trail completeness for automated processing

Output ONLY valid JSON, no markdown fences:

{
  "task": "A",
  "gaps": [
    {
      "finding": "one sentence — the specific gap",
      "severity": "HIGH|MEDIUM|LOW",
      "evidence": "quote from doc or file:line that proves it",
      "recommendation": "one sentence fix"
    }
  ]
}

Maximum 7 gaps. Rank by severity descending.
"""

SYSTEM_B = """\
You are a Senior Engineer Auditor reviewing a software project.

Your job: find gaps between what the project claims to have built (in documentation and commit history)
and what is actually implemented in the code.
Cross-reference documentation against the codebase. Be specific.

Focus on:
- Features described in docs/brief that have no corresponding implementation
- Missing engineering safeguards (versioning, pinning, concurrency, error handling)
- Operational risks that are documented scenarios but have no code addressing them
- Configuration that uses mutable identifiers where immutable ones are needed

Output ONLY valid JSON, no markdown fences:

{
  "task": "B",
  "gaps": [
    {
      "finding": "one sentence — the specific gap",
      "severity": "HIGH|MEDIUM|LOW",
      "evidence": "quote from doc or file:line that proves it",
      "recommendation": "one sentence fix"
    }
  ]
}

Maximum 7 gaps. Rank by severity descending.
"""


# ── Inference ─────────────────────────────────────────────────────────────────

def call_ollama(system: str, context: str, model: str) -> tuple[str, float, dict]:
    import requests
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json={
        "model":   model,
        "system":  system,
        "prompt":  context + "\n\nNow output the JSON gaps array.",
        "stream":  False,
        "think":   False,
        "options": {"temperature": 0.1, "num_predict": 2048},
    }, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    input_tok  = data.get("prompt_eval_count", 0)
    output_tok = data.get("eval_count", 0)
    eval_ns    = data.get("eval_duration", 0)
    tps        = output_tok / (eval_ns / 1e9) if eval_ns > 0 else 0
    stats = {"input_tokens": input_tok, "output_tokens": output_tok,
             "total_tokens": input_tok + output_tok, "tokens_per_s": round(tps, 1)}
    return data["response"].strip(), round(time.time() - t0, 2), stats


def call_api(system: str, context: str, model: str) -> tuple[str, float, dict]:
    try:
        import anthropic
    except ImportError:
        print("[ERROR] anthropic package not installed: pip install anthropic", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic()
    t0 = time.time()
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": context + "\n\nNow output the JSON gaps array."}],
    )
    duration = round(time.time() - t0, 2)
    raw = msg.content[0].text.strip()
    stats = {
        "input_tokens":  msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "total_tokens":  msg.usage.input_tokens + msg.usage.output_tokens,
        "tokens_per_s":  None,
    }
    return raw, duration, stats


def parse_output(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ── Run one task ──────────────────────────────────────────────────────────────

def run_task(task_id: str, system: str, context: str,
             model: str, use_api: bool, reps: int) -> list[dict]:
    results = []
    for rep in range(1, reps + 1):
        if rep > 1:
            print(f"      idle {IDLE_BETWEEN_REPS}s...", end=" ", flush=True)
            time.sleep(IDLE_BETWEEN_REPS)
        print(f"    rep {rep}/{reps}... ", end="", flush=True)
        try:
            if use_api:
                raw, duration, stats = call_api(system, context, model)
            else:
                raw, duration, stats = call_ollama(system, context, model)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"rep": rep, "error": str(e)})
            continue

        try:
            parsed = parse_output(raw)
        except json.JSONDecodeError as e:
            print(f"JSON parse failed ({e})")
            parsed = {"raw": raw, "parse_error": str(e)}

        n_gaps = len(parsed.get("gaps", []))
        print(f"{duration:.1f}s  {stats['input_tokens']}→{stats['output_tokens']} tok  {n_gaps} gaps")
        results.append({"rep": rep, "duration_s": duration, "tokens": stats, **parsed})

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project",  "-p", required=True)
    ap.add_argument("--model",    "-m", default="gemma4:26b")
    ap.add_argument("--api",            action="store_true", help="Use Anthropic API instead of Ollama")
    ap.add_argument("--dry-run",  "-n", action="store_true")
    ap.add_argument("--reps",           type=int, default=REPS)
    args = ap.parse_args()

    project = Path(args.project).expanduser().resolve()
    if not project.is_dir():
        print(f"[ERROR] Not a directory: {project}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Exp 012 — {args.model} ({'API' if args.api else 'Ollama'}) ===")
    print(f"  Project: {project}")
    print(f"  Reps: {args.reps}  |  Tasks: A (DPO) + B (Engineer)")

    ctx_a = context_task_a(project)
    ctx_b = context_task_b(project)

    print(f"  Context A: {len(ctx_a):,} chars  |  Context B: {len(ctx_b):,} chars")

    if args.dry_run:
        print("\n--- TASK A context (first 2000 chars) ---")
        print(ctx_a[:2000])
        print("\n--- TASK B context (first 2000 chars) ---")
        print(ctx_b[:2000])
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = args.model.replace(":", "_").replace("-", "_").replace(".", "_")

    print("\n  [TASK A — DPO Compliance]")
    reps_a = run_task("A", SYSTEM_A, ctx_a, args.model, args.api, args.reps)

    print("\n  [TASK B — Implementation Gaps]")
    reps_b = run_task("B", SYSTEM_B, ctx_b, args.model, args.api, args.reps)

    record = {
        "experiment": "012",
        "ts": ts,
        "model": args.model,
        "runtime": "api" if args.api else "ollama",
        "project": str(project),
        "reps_per_task": args.reps,
        "task_a": reps_a,
        "task_b": reps_b,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{model_slug}_{ts}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\n  Saved → {out.name}")

    # Summary
    print("\n  --- Task A gaps (rep 1) ---")
    rep1_a = reps_a[0] if reps_a else {}
    for g in rep1_a.get("gaps", []):
        sev = g.get("severity", "?")
        print(f"    [{sev}] {g.get('finding', '')[:90]}")

    print("\n  --- Task B gaps (rep 1) ---")
    rep1_b = reps_b[0] if reps_b else {}
    for g in rep1_b.get("gaps", []):
        sev = g.get("severity", "?")
        print(f"    [{sev}] {g.get('finding', '')[:90]}")

    print()


if __name__ == "__main__":
    main()
