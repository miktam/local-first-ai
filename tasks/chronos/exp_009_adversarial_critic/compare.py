#!/usr/bin/env python3
"""
Experiment 009 — Comparison Harness

Compares a Claude result and a gemma4 result from the results/ directory.
Outputs overlap metrics to be scored by a human reviewer.

Usage:
  python3 compare.py                          # compare most recent pair
  python3 compare.py --claude FILE --gemma FILE
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_latest(prefix: str) -> tuple[Path, dict]:
    files = sorted(RESULTS_DIR.glob(f"{prefix}*.json"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No {prefix}*.json in {RESULTS_DIR}")
    path = files[0]
    return path, json.loads(path.read_text(encoding="utf-8"))


def all_issues(result: dict) -> list[str]:
    issues = []
    for persona in result.get("personas", {}).values():
        issues.extend(persona.get("weak", []))
        issues.extend(persona.get("missing", []))
    return issues


def print_comparison(claude: dict, gemma: dict) -> None:
    c_issues = all_issues(claude)
    g_issues = all_issues(gemma)

    def fmt_tokens(r: dict) -> str:
        t = r.get("tokens", {})
        if not t:
            chars = r.get("prompt_chars", 0)
            est = chars // 4
            return f"~{est} in (est) · output: unknown  [Claude — estimate only]"
        return (f"{t.get('input_tokens','?')} in · "
                f"{t.get('output_tokens','?')} out · "
                f"{t.get('total_tokens','?')} total · "
                f"{t.get('tokens_per_s','?')} tok/s")

    print(f"\n{'='*60}")
    print(f"EXPERIMENT 009 — ADVERSARIAL CRITIC COMPARISON")
    print(f"{'='*60}")
    print(f"\nClaude  ({claude.get('critic','?')})")
    print(f"  issues: {len(c_issues)}  time: {claude.get('duration_s','?')}s")
    print(f"  tokens: {fmt_tokens(claude)}")
    print(f"\ngemma4  ({gemma.get('critic','?')})")
    print(f"  issues: {len(g_issues)}  time: {gemma.get('duration_s','?')}s")
    print(f"  tokens: {fmt_tokens(gemma)}")
    print()

    for persona in ("investor", "dpo", "engineer"):
        c = claude.get("personas", {}).get(persona, {})
        g = gemma.get("personas", {}).get(persona, {})
        print(f"── {persona.upper()} ──")
        print(f"  Claude  weak:    {c.get('weak', [])}")
        print(f"  gemma4  weak:    {g.get('weak', [])}")
        print(f"  Claude  missing: {c.get('missing', [])}")
        print(f"  gemma4  missing: {g.get('missing', [])}")
        print()

    print("── TOP ACTIONS ──")
    print(f"  Claude: {claude.get('top_actions', [])}")
    print(f"  gemma4: {gemma.get('top_actions', [])}")

    print(f"\n── SEVERITY ──")
    print(f"  Claude: {claude.get('severity_counts', {})}")
    print(f"  gemma4: {gemma.get('severity_counts', {})}")

    print(f"\n{'='*60}")
    print("SCORING (human reviewer fills this in)")
    print(f"{'='*60}")
    print(f"Total Claude high-severity issues: _____")
    print(f"gemma4 matched (same issue, different words OK): _____")
    print(f"Overlap rate: _____% (target: ≥60%)")
    print(f"gemma4 false positives (not in Claude): _____")
    print(f"False positive rate: _____% (target: ≤30%)")
    print(f"\nVerdict: PASS / FAIL")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Exp 009 — Compare Claude vs gemma4 critic")
    parser.add_argument("--claude", help="Path to Claude result JSON")
    parser.add_argument("--gemma",  help="Path to gemma4 result JSON")
    args = parser.parse_args()

    if args.claude:
        claude_path = Path(args.claude)
        claude = json.loads(claude_path.read_text(encoding="utf-8"))
    else:
        claude_path, claude = load_latest("claude")
        print(f"  Claude result: {claude_path.name}")

    if args.gemma:
        gemma_path = Path(args.gemma)
        gemma = json.loads(gemma_path.read_text(encoding="utf-8"))
    else:
        gemma_path, gemma = load_latest("gemma4")
        print(f"  gemma4 result: {gemma_path.name}")

    print_comparison(claude, gemma)


if __name__ == "__main__":
    main()
