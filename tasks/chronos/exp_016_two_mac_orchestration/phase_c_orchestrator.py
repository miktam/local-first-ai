#!/usr/bin/env python3
"""
Exp 016 Phase C — Tiered pipeline end-to-end orchestrator.

Task (frozen before first run):
  Add a `price_per_sqm` computed field to CasaSol's get_property response.
  Field = price / size_m2, rounded to nearest integer.
  Null if either field is absent or zero.
  New line: **Price per sqm:** €X,XXX/m²  (omitted if null)
  Add regression test in tests/test_mcp_server.py.

Design:
  BASELINE — single gemma4:26b call (mini Ollama) with full task, no pre-plan.
  TIERED   — Qwen3-Coder-Next (MBP MLX) produces plan.json first,
              then gemma4:26b executes each step from plan.json.

Measurement:
  intervention: a step whose output required a retry or manual correction.
  Gate (H5): tiered interventions < baseline interventions.

Writes:
  measurements/phase_c_tiered_pipeline.json

Run from exp_016 directory:
  python3 phase_c_orchestrator.py
"""

import json
import os
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ─── endpoints ────────────────────────────────────────────────────────────────
CHEAP_URL   = "http://localhost:11434/v1/chat/completions"
CHEAP_MODEL = "gemma4:26b"

SMART_URL   = "http://miktam-mbp.local:8080/v1/chat/completions"
SMART_MODEL = "mlx-community/Qwen3-Coder-Next-4bit"

# ─── paths ────────────────────────────────────────────────────────────────────
CASASOL = Path("/Users/miktam02/REPOS/casasol")
MCP_FILE  = CASASOL / "scripts/mcp_server.py"
TEST_FILE = CASASOL / "tests/test_mcp_server.py"
OUT_JSON  = Path("measurements/phase_c_tiered_pipeline.json")

TIMEOUT_S = 120

# ─── task context (frozen) ────────────────────────────────────────────────────
TASK_DESC = """\
Add a `price_per_sqm` computed field to the CasaSol get_property response.

Rules:
- Compute price_per_sqm = round(price / size_m2) where both price (INTEGER, EUR) \
and size_m2 (REAL, m²) exist and are non-zero.
- If either is absent or zero, omit the line entirely.
- Add the field as a new line in format_listing_full(), immediately after the \
existing **Price:** line:
  **Price per sqm:** €{price_per_sqm:,}/m²
- Add a regression test in tests/test_mcp_server.py called \
test_get_property_price_per_sqm that verifies:
  (a) when price=1000000 and size_m2=300, the string "3,333" appears in the result
  (b) when size_m2 is 0 or absent, the "Price per sqm" line does not appear
"""

# ─── API call ─────────────────────────────────────────────────────────────────

def call_model(url: str, model: str, messages: list, label: str = "") -> dict:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
        "stream": False,
        "thinking": False,
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            raw = json.loads(resp.read())
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elapsed_s": round(time.perf_counter() - t0, 2)}

    elapsed = round(time.perf_counter() - t0, 2)
    choices = raw.get("choices", [])
    if not choices:
        return {"ok": False, "error": "empty choices", "raw": raw, "elapsed_s": elapsed}

    msg = choices[0].get("message", {})
    content = msg.get("content") or msg.get("reasoning") or ""
    print(f"  [{label}] {elapsed}s  {len(content)} chars")
    return {"ok": True, "content": content, "elapsed_s": elapsed, "model": model}


# ─── code extraction ──────────────────────────────────────────────────────────

def extract_fence(text: str, lang: str = "python") -> str | None:
    """Extract content of the first code fence of the given language."""
    pattern = rf"```{lang}\s*\n(.*?)```"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    # fallback: any fence
    m = re.search(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else None


def extract_json_fence(text: str) -> dict | None:
    code = extract_fence(text, "json")
    if not code:
        # try bare JSON
        m = re.search(r'\{.*"steps".*\}', text, re.DOTALL)
        code = m.group(0) if m else None
    if not code:
        return None
    try:
        return json.loads(code)
    except json.JSONDecodeError:
        return None


# ─── file operations ─────────────────────────────────────────────────────────

def read_section(path: Path, start_marker: str, end_marker: str = None) -> str:
    """Read lines from start_marker to end_marker (exclusive), or to EOF."""
    text = path.read_text()
    if start_marker not in text:
        return ""
    start = text.index(start_marker)
    if end_marker and end_marker in text[start + 1:]:
        end = text.index(end_marker, start + 1)
        return text[start:end]
    return text[start:]


def apply_patch(path: Path, old: str, new: str) -> bool:
    """Replace old with new in path. Returns True if applied."""
    content = path.read_text()
    if old not in content:
        return False
    path.write_text(content.replace(old, new, 1))
    return True


def run_tests() -> dict:
    """Run CasaSol test suite, return pass/fail counts."""
    result = subprocess.run(
        [str(CASASOL / ".venv/bin/pytest"), "tests/test_mcp_server.py", "-v", "--tb=short", "-q"],
        capture_output=True, text=True, cwd=CASASOL, timeout=60
    )
    output = result.stdout + result.stderr
    m_pass  = re.search(r"(\d+) passed", output)
    m_fail  = re.search(r"(\d+) failed", output)
    m_err   = re.search(r"(\d+) error",  output)
    passed  = int(m_pass.group(1))  if m_pass  else 0
    failed  = int(m_fail.group(1))  if m_fail  else 0
    errors  = int(m_err.group(1))   if m_err   else 0
    return {
        "returncode": result.returncode,
        "passed": passed, "failed": failed, "errors": errors,
        "output": output[-3000:],
    }


def save_state(paths: list[Path]) -> dict:
    return {str(p): p.read_text() for p in paths}


def restore_state(saved: dict):
    for path, content in saved.items():
        Path(path).write_text(content)


# ─── BASELINE run ─────────────────────────────────────────────────────────────

def run_baseline(log: dict) -> int:
    """Single gemma4:26b call with full task. Returns intervention count (0 or 1)."""
    print("\n" + "="*60)
    print("BASELINE — single gemma4:26b call, no plan")
    print("="*60)

    saved = save_state([MCP_FILE, TEST_FILE])

    format_listing_full_src = read_section(MCP_FILE, "def format_listing_full(")
    test_fixture_src = read_section(TEST_FILE, "@pytest.fixture\ndef setup_test_sqlite")

    messages = [
        {"role": "user", "content": f"""\
{TASK_DESC}

Here is the current format_listing_full function from scripts/mcp_server.py:

```python
{format_listing_full_src}
```

Here is the test fixture from tests/test_mcp_server.py:

```python
{test_fixture_src}
```

Produce:
1. The complete replacement for format_listing_full (full function, no ellipsis).
2. A new test function test_get_property_price_per_sqm to append to test_mcp_server.py.

Output two python code fences: first the replacement function, second the test.
"""}
    ]

    r = call_model(CHEAP_URL, CHEAP_MODEL, messages, "baseline/cheap")
    log["baseline"] = {"call": r}

    if not r["ok"]:
        log["baseline"]["verdict"] = "CALL_FAILED"
        log["baseline"]["interventions"] = 1
        restore_state(saved)
        return 1

    content = r["content"]
    fences = re.findall(r"```python\s*\n(.*?)```", content, re.DOTALL | re.IGNORECASE)

    impl_code = fences[0] if fences else None
    test_code  = fences[1] if len(fences) > 1 else None

    applied_impl = applied_test = False

    if impl_code and "def format_listing_full" in impl_code:
        old_fn = read_section(MCP_FILE, "def format_listing_full(", "\n\n@")
        if not old_fn:
            old_fn = read_section(MCP_FILE, "def format_listing_full(", "\n# ---")
        if apply_patch(MCP_FILE, old_fn, impl_code):
            applied_impl = True
            print("  [baseline] Applied format_listing_full patch")

    if test_code and "def test_get_property_price_per_sqm" in test_code:
        existing = TEST_FILE.read_text()
        if "test_get_property_price_per_sqm" not in existing:
            TEST_FILE.write_text(existing + "\n\n" + test_code)
            applied_test = True
            print("  [baseline] Appended test function")

    test_result = run_tests()
    log["baseline"]["applied_impl"] = applied_impl
    log["baseline"]["applied_test"] = applied_test
    log["baseline"]["test_result"] = test_result

    if test_result["returncode"] == 0:
        print(f"  [baseline] Tests PASSED ({test_result['passed']} passed)")
        log["baseline"]["verdict"] = "PASSED"
        log["baseline"]["interventions"] = 0
        # keep changes — this is now the implementation
        return 0
    else:
        print(f"  [baseline] Tests FAILED ({test_result['failed']} failed, {test_result['errors']} errors)")
        log["baseline"]["verdict"] = "FAILED"
        log["baseline"]["interventions"] = 1
        restore_state(saved)
        return 1


# ─── TIERED run ───────────────────────────────────────────────────────────────

def run_tiered(log: dict) -> int:
    """Smart plan → cheap execute. Returns total intervention count."""
    print("\n" + "="*60)
    print("TIERED — Qwen3-Coder-Next plans, gemma4:26b executes")
    print("="*60)

    saved = save_state([MCP_FILE, TEST_FILE])
    interventions = 0
    steps_log = []

    # ── Step 1: smart tier produces plan.json ──────────────────────────────
    print("\n  [smart] Planning...")
    format_listing_full_src = read_section(MCP_FILE, "def format_listing_full(")
    test_fixture_src = read_section(TEST_FILE, "@pytest.fixture\ndef setup_test_sqlite")

    plan_messages = [{"role": "user", "content": f"""\
You are a planning agent. Produce a JSON execution plan for the following task.
Output ONLY a JSON code fence. No prose before or after.

Task:
{TASK_DESC}

Relevant code context:

format_listing_full in scripts/mcp_server.py:
```python
{format_listing_full_src}
```

Test fixture in tests/test_mcp_server.py:
```python
{test_fixture_src}
```

Output this exact schema (no other keys):
{{
  "task": "<one-line summary>",
  "steps": [
    {{
      "id": "step-01",
      "role": "cheap",
      "instruction": "<precise instruction for the cheap model — include the exact function name and what to change>",
      "context_budget_tokens": 1500
    }}
  ]
}}

Split into 2 steps: one for modifying format_listing_full, one for adding the test.
"""}]

    plan_r = call_model(SMART_URL, SMART_MODEL, plan_messages, "smart/plan")
    log["tiered"] = {"plan_call": plan_r, "steps": []}

    plan = None
    if plan_r["ok"]:
        plan = extract_json_fence(plan_r["content"])

    if not plan or "steps" not in plan:
        print("  [smart] Failed to produce valid plan.json — intervention")
        log["tiered"]["plan_verdict"] = "FAILED"
        restore_state(saved)
        return 99

    # Commit plan.json to disk before cheap starts (deterministic-glue contract)
    plan_path = Path("measurements/phase_c_plan.json")
    plan_path.write_text(json.dumps(plan, indent=2))
    print(f"  [smart] plan.json written ({len(plan['steps'])} steps): {plan_path}")
    log["tiered"]["plan"] = plan
    log["tiered"]["plan_path"] = str(plan_path)

    # ── Step 2: cheap tier executes each step ─────────────────────────────
    for step in plan["steps"]:
        step_id   = step.get("id", "?")
        role      = step.get("role", "cheap")
        instr     = step.get("instruction", "")
        budget    = step.get("context_budget_tokens", 1500)

        print(f"\n  [cheap/{step_id}] role={role}  budget={budget}t")

        # Cheap model gets only: step instruction + current relevant file snippet
        if "format_listing_full" in instr or "mcp_server" in instr:
            snippet = read_section(MCP_FILE, "def format_listing_full(")
            file_hint = "scripts/mcp_server.py"
        else:
            snippet = TEST_FILE.read_text()[-3000:]
            file_hint = "tests/test_mcp_server.py"

        step_messages = [{"role": "user", "content": f"""\
{instr}

Current code in {file_hint}:
```python
{snippet}
```

Output the complete replacement code in a single python code fence.
Do not omit any lines. Do not use ellipsis. Do not add commentary outside the fence.
"""}]

        step_r = call_model(CHEAP_URL, CHEAP_MODEL, step_messages, f"cheap/{step_id}")
        step_log = {"step": step, "call": step_r}

        applied = False
        if step_r["ok"]:
            content = step_r["content"]
            code = extract_fence(content, "python")
            if code:
                if "def format_listing_full" in code:
                    old = read_section(MCP_FILE, "def format_listing_full(", "\n\n@")
                    if not old:
                        old = read_section(MCP_FILE, "def format_listing_full(", "\n# ---")
                    if apply_patch(MCP_FILE, old, code):
                        applied = True
                        print(f"  [cheap/{step_id}] Applied mcp_server patch")
                elif "def test_get_property_price_per_sqm" in code:
                    existing = TEST_FILE.read_text()
                    if "test_get_property_price_per_sqm" not in existing:
                        TEST_FILE.write_text(existing + "\n\n" + code)
                        applied = True
                        print(f"  [cheap/{step_id}] Appended test function")

        step_log["applied"] = applied
        if not applied:
            print(f"  [cheap/{step_id}] Could not apply output — intervention")
            interventions += 1
            step_log["intervention"] = True

        steps_log.append(step_log)

    log["tiered"]["steps"] = steps_log

    # ── Final test run ─────────────────────────────────────────────────────
    test_result = run_tests()
    log["tiered"]["test_result"] = test_result
    log["tiered"]["interventions"] = interventions

    if test_result["returncode"] == 0:
        print(f"\n  [tiered] Tests PASSED ({test_result['passed']} passed)")
        log["tiered"]["verdict"] = "PASSED"
    else:
        print(f"\n  [tiered] Tests FAILED ({test_result['failed']} failed, {test_result['errors']} errors)")
        log["tiered"]["verdict"] = "FAILED"
        interventions += 1
        log["tiered"]["interventions"] = interventions
        restore_state(saved)

    return interventions


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\nExp 016 Phase C — Tiered pipeline end-to-end")
    print(f"Started: {run_ts}")
    print(f"Task: {TASK_DESC.splitlines()[0]}")

    log = {"experiment": "exp_016_phase_c", "run_ts": run_ts, "task": TASK_DESC}

    baseline_interventions = run_baseline(log)
    baseline_verdict = log["baseline"]["verdict"]

    # Restore files to original state before tiered run
    # (if baseline passed, we need to undo it to test tiered independently)
    saved_original = save_state([MCP_FILE, TEST_FILE])
    # Actually: read original from git
    for fpath in [MCP_FILE, TEST_FILE]:
        result = subprocess.run(
            ["git", "show", f"HEAD:{fpath.relative_to(CASASOL)}"],
            capture_output=True, text=True, cwd=CASASOL
        )
        if result.returncode == 0:
            fpath.write_text(result.stdout)
    print("\n  [orchestrator] Restored original files from git HEAD before tiered run")

    tiered_interventions = run_tiered(log)
    tiered_verdict = log["tiered"]["verdict"]

    # ── Comparison ─────────────────────────────────────────────────────────
    h5_confirmed = tiered_interventions < baseline_interventions
    log["comparison"] = {
        "baseline_interventions": baseline_interventions,
        "baseline_verdict": baseline_verdict,
        "tiered_interventions": tiered_interventions,
        "tiered_verdict": tiered_verdict,
        "h5_result": "H5_CONFIRMED" if h5_confirmed else "H5_FALSIFIED",
    }

    OUT_JSON.write_text(json.dumps(log, indent=2))

    print("\n" + "="*60)
    print("  PHASE C RESULTS")
    print("="*60)
    print(f"  Baseline:  {baseline_verdict}  ({baseline_interventions} interventions)")
    print(f"  Tiered:    {tiered_verdict}  ({tiered_interventions} interventions)")
    print(f"  H5:        {'CONFIRMED' if h5_confirmed else 'FALSIFIED'}")
    print(f"  Written:   {OUT_JSON}")

    # If tiered produced a working implementation, apply is already live.
    # If baseline produced a working implementation (and tiered didn't), restore baseline.
    if baseline_verdict == "PASSED" and tiered_verdict != "PASSED":
        print("  [orchestrator] Tiered didn't pass tests; restoring baseline implementation.")
        for path, content in saved_original.items():
            Path(path).write_text(content)


if __name__ == "__main__":
    main()
