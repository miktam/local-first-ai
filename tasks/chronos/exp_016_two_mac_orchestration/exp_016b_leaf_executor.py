#!/usr/bin/env python3
"""
Exp 016B — Local LLM as leaf executor in a problem decomposition tree.

Demonstrates the deterministic-glue architecture:
  - Smart tier (MBP, Qwen3-Coder-Next): decomposes the task into a tree of leaves.
  - Orchestrator (deterministic Python): traverses the tree, does all joins.
  - Cheap tier (mini, gemma4:26b): executes each leaf with a STRICT output contract.

Leaf contract:
  INPUT  — one code snippet (the exact target) + one precise instruction
  OUTPUT — one Python code fence, nothing else
  Validated before apply. Retried once on failure. Intervention counted on second failure.

No leaf receives another leaf's prose. Leaf B's context includes Leaf A's output as a
code fence — the orchestrator does the injection in deterministic code.

Task (frozen before first run):
  Add `price_tier` classification to CasaSol.
  - _price_tier(price) utility: budget/mid/premium/luxury/None
  - format_listing_full: show **Price tier:** line
  - search_properties: accept price_tier= filter
  - tests: utility + display + filter

Tree:
  root (smart): produce LeafSpec tree
  ├── leaf-A (cheap): write _price_tier() utility function
  ├── leaf-B (cheap): modify format_listing_full to include tier  [deps: leaf-A]
  ├── leaf-C (cheap): add price_tier param+filter to search_properties [deps: none]
  └── leaf-D (cheap): write 3 regression tests                   [deps: leaf-A]

Orchestrator joins: after leaf-A, inject its output (code fence) into leaf-B and leaf-D
context. No model call is involved in the join.

Writes:
  measurements/exp_016b_tree.json      — the tree produced by the smart tier
  measurements/exp_016b_results.json   — full run log (leaf outputs, test results)

Run from exp_016 directory:
  python3 exp_016b_leaf_executor.py
"""

import ast
import json
import re
import subprocess
import sys
import textwrap
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
CASASOL   = Path("/Users/miktam02/REPOS/casasol")
MCP_FILE  = CASASOL / "scripts/mcp_server.py"
TEST_FILE = CASASOL / "tests/test_mcp_server.py"
PYTEST    = CASASOL / ".venv/bin/pytest"
OUT_TREE  = Path("measurements/exp_016b_tree.json")
OUT_LOG   = Path("measurements/exp_016b_results.json")

TIMEOUT_S = 120
FORCE_FALLBACK = True  # set True to skip smart tier and use pre-designed fallback tree

# ─── task context (frozen before run) ─────────────────────────────────────────
TASK_DESC = """\
Add `price_tier` classification to CasaSol (casasol/scripts/mcp_server.py).

Price tier rules (EUR):
  budget   → price < 300,000
  mid      → 300,000 ≤ price < 800,000
  premium  → 800,000 ≤ price < 2,000,000
  luxury   → price ≥ 2,000,000
  None     → price is None, 0, or absent

Changes required:
  1. New utility function _price_tier(price: int | None) -> str | None
     Insert before the format_listing() function.
  2. In format_listing_full(), add "**Price tier:** {tier}" line after "**Price per sqm:**".
     Only add the line if price_tier is not None.
  3. In search_properties(), add a price_tier: str = "" parameter.
     Add SQLite filter: if price_tier is set, add a WHERE condition using a Python
     helper that maps the tier name back to the price range.
  4. Regression tests in tests/test_mcp_server.py:
     (a) test that _price_tier(250000) == "budget", _price_tier(1000000) == "premium",
         _price_tier(None) is None
     (b) test that get_property for a listing with price=1000000 includes "Price tier"
     (c) test that search_properties(price_tier="budget") filters correctly
"""

# ─── fallback tree (used if smart tier fails to produce valid JSON) ───────────
FALLBACK_TREE = {
    "task": "Add price_tier classification to CasaSol",
    "leaves": [
        {
            "id": "leaf-A",
            "deps": [],
            "file": "scripts/mcp_server.py",
            "target_symbol": "_price_tier",
            "instruction": (
                "Write a new Python function `_price_tier(price: int | None) -> str | None` "
                "that returns 'budget' for price<300000, 'mid' for 300000<=price<800000, "
                "'premium' for 800000<=price<2000000, 'luxury' for price>=2000000, "
                "and None if price is None or 0. "
                "Output the complete function only."
            ),
        },
        {
            "id": "leaf-B",
            "deps": ["leaf-A"],
            "file": "scripts/mcp_server.py",
            "target_symbol": "format_listing_full",
            "instruction": (
                "Modify format_listing_full() to call _price_tier(row.get('price')) "
                "and add a '**Price tier:** {tier}' line immediately after the "
                "'**Price per sqm:**' line. Omit the line if tier is None. "
                "Output the complete replacement function only."
            ),
        },
        {
            "id": "leaf-C",
            "deps": [],
            "file": "scripts/mcp_server.py",
            "target_symbol": "search_properties",
            "instruction": (
                "Add a `price_tier: str = ''` parameter to search_properties(). "
                "When price_tier is set, add a SQLite WHERE condition using only "
                "standard Python comparisons (no external imports, no undefined names). "
                "Use inline if/elif: "
                "if price_tier == 'budget': conditions.append('price < ?'); params.append(300000). "
                "elif price_tier == 'mid': conditions.append('price >= ? AND price < ?'); params.extend([300000,800000]). "
                "elif price_tier == 'premium': conditions.append('price >= ? AND price < ?'); params.extend([800000,2000000]). "
                "elif price_tier == 'luxury': conditions.append('price >= ?'); params.append(2000000). "
                "Add this block in the SQLite path after the existing max_price block. "
                "Also add price_tier to the ChromaDB where-clause metadata filter when set, "
                "using the same if/elif pattern appending to clauses. "
                "Output the complete replacement function only."
            ),
        },
        {
            "id": "leaf-D1",
            "deps": ["leaf-A"],
            "file": "tests/test_mcp_server.py",
            "target_symbol": "test_price_tier_utility",
            "instruction": (
                "Write one pytest function `test_price_tier_utility` that imports "
                "_price_tier from scripts.mcp_server and asserts: "
                "_price_tier(250000)=='budget', _price_tier(1000000)=='premium', "
                "_price_tier(5000000)=='luxury', _price_tier(None) is None, _price_tier(0) is None. "
                "Output only that one function in a code fence."
            ),
        },
        {
            "id": "leaf-D2",
            "deps": ["leaf-A", "leaf-B"],
            "file": "tests/test_mcp_server.py",
            "target_symbol": "test_get_property_price_tier",
            "instruction": (
                "Write one pytest function `test_get_property_price_tier(setup_test_sqlite)` "
                "that patches 'scripts.mcp_server.get_db' with setup_test_sqlite, "
                "calls mcp_server.get_property(casasol_id='test-001') (price=1000000), "
                "and asserts 'Price tier' in result and 'premium' in result. "
                "Use the same patch pattern as the other tests in the file. "
                "Output only that one function in a code fence."
            ),
        },
        {
            "id": "leaf-D3",
            "deps": ["leaf-C"],
            "file": "tests/test_mcp_server.py",
            "target_symbol": "test_search_properties_price_tier",
            "instruction": (
                "Write one pytest function `test_search_properties_price_tier(setup_test_sqlite)` "
                "that patches 'scripts.mcp_server.get_db' with setup_test_sqlite, "
                "calls mcp_server.search_properties(price_tier='budget'), "
                "and asserts 'No properties found' in result "
                "(all fixture prices are >=500000, none qualify as budget <300000). "
                "Use the same patch pattern as the other tests in the file. "
                "Output only that one function in a code fence."
            ),
        },
    ],
}


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def call_model(url: str, model: str, messages: list, label: str = "", timeout: int = TIMEOUT_S) -> dict:
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read())
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elapsed_s": round(time.perf_counter() - t0, 2)}

    elapsed = round(time.perf_counter() - t0, 2)
    msg = (raw.get("choices") or [{}])[0].get("message", {})
    content = msg.get("content") or msg.get("reasoning") or ""
    tokens  = raw.get("usage", {}).get("completion_tokens", 0)
    print(f"  [{label}] {elapsed}s  {tokens} tokens  {len(content)} chars")
    return {"ok": True, "content": content, "elapsed_s": elapsed, "tokens": tokens}


# ─── strict leaf output contract ──────────────────────────────────────────────
SYSTEM_LEAF = (
    "You are a code transducer. "
    "Your ENTIRE response must be a single Python code fence: ```python ... ```. "
    "No prose. No reasoning. No explanation. No text before or after the fence. "
    "If you cannot complete the task, output an empty code fence: ```python\n```"
)


def extract_code_fence(text: str) -> str | None:
    """Return content of the first python code fence, or None."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else None


def validate_code(code: str, must_contain: str) -> tuple[bool, str]:
    """Check that code is valid Python and contains must_contain."""
    if not code or not code.strip():
        return False, "empty"
    normalized = textwrap.dedent(code)
    try:
        ast.parse(normalized)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    if must_contain not in normalized:
        return False, f"missing symbol: {must_contain!r}"
    return True, "ok"


def leaf_call(leaf_id: str, instruction: str, context_snippets: list[str], must_contain: str) -> dict:
    """
    Call the cheap tier with strict leaf contract. Retry once on validation failure.
    Returns: {"code": str | None, "interventions": int, "calls": list}
    """
    ctx_block = "\n\n".join(
        f"```python\n{s}\n```" for s in context_snippets if s.strip()
    )
    user_msg = f"{instruction}\n\nRelevant code:\n{ctx_block}\n\nOUTPUT: A SINGLE PYTHON CODE FENCE. NOTHING ELSE."

    messages = [
        {"role": "system", "content": SYSTEM_LEAF},
        {"role": "user",   "content": user_msg},
    ]

    calls = []
    for attempt in range(2):
        label = f"cheap/{leaf_id}" if attempt == 0 else f"cheap/{leaf_id}/retry"
        if attempt == 1:
            messages.append({"role": "assistant", "content": calls[-1]["raw"]})
            messages.append({"role": "user", "content":
                "WRONG FORMAT. Respond with ONLY a ```python...``` code fence. Nothing else."})

        r = call_model(CHEAP_URL, CHEAP_MODEL, messages, label)
        raw_content = r.get("content", "") if r["ok"] else ""
        code = extract_code_fence(raw_content) if raw_content else None
        ok, reason = validate_code(code, must_contain) if code else (False, "no fence")

        calls.append({"attempt": attempt, "ok": r["ok"], "valid": ok,
                      "reason": reason, "elapsed_s": r.get("elapsed_s"),
                      "raw": raw_content[:500]})

        if ok:
            code = textwrap.dedent(code)  # normalize before returning; validate used dedented copy
            print(f"  [{leaf_id}] attempt={attempt} VALID — {must_contain!r} found")
            return {"code": code, "interventions": attempt, "calls": calls}

        print(f"  [{leaf_id}] attempt={attempt} INVALID — {reason}")

    print(f"  [{leaf_id}] FAILED after 2 attempts — INTERVENTION")
    return {"code": None, "interventions": 1, "calls": calls}


# ─── file operations ──────────────────────────────────────────────────────────

def read_function(path: Path, fn_name: str) -> str:
    """Read from 'def fn_name' to the next top-level def/class or EOF."""
    text = path.read_text()
    pattern = rf"(^|\n)(def {re.escape(fn_name)}[\s(])"
    m = re.search(pattern, text)
    if not m:
        return ""
    start = m.start(2) if m.group(1) == "\n" else m.start()
    # Find next top-level def/class
    rest = text[start:]
    next_top = re.search(r"\n(def |class |@)", rest[1:])
    if next_top:
        return rest[:next_top.start() + 1]
    return rest


def read_last_n(path: Path, n: int = 60) -> str:
    """Read last n lines of a file."""
    lines = path.read_text().splitlines()
    return "\n".join(lines[-n:])


def apply_replacement(path: Path, old_fn_name: str, new_code: str) -> bool:
    """Replace the old function body with new_code. Returns True on success."""
    old = read_function(path, old_fn_name)
    if not old:
        return False
    content = path.read_text()
    if old not in content:
        return False
    path.write_text(content.replace(old, new_code, 1))
    return True


def insert_before(path: Path, marker_fn: str, new_code: str) -> bool:
    """Insert new_code immediately before 'def marker_fn'. Returns True on success."""
    text = path.read_text()
    pattern = rf"(^|\n)(def {re.escape(marker_fn)}[\s(])"
    m = re.search(pattern, text)
    if not m:
        return False
    pos = m.start(2) if m.group(1) == "\n" else m.start()
    text = text[:pos] + new_code.rstrip() + "\n\n\n" + text[pos:]
    path.write_text(text)
    return True


def append_tests(path: Path, new_code: str) -> bool:
    """Append new_code at module level to the test file."""
    new_code = textwrap.dedent(new_code).strip()
    existing = path.read_text()
    path.write_text(existing.rstrip() + "\n\n\n" + new_code + "\n")
    return True


def run_tests() -> dict:
    result = subprocess.run(
        [str(PYTEST), "tests/test_mcp_server.py", "-v", "--tb=short"],
        capture_output=True, text=True, cwd=CASASOL, timeout=60,
    )
    out = result.stdout + result.stderr
    m_pass = re.search(r"(\d+) passed", out)
    m_fail = re.search(r"(\d+) failed", out)
    m_err  = re.search(r"(\d+) error",  out)
    return {
        "returncode": result.returncode,
        "passed":  int(m_pass.group(1)) if m_pass else 0,
        "failed":  int(m_fail.group(1)) if m_fail else 0,
        "errors":  int(m_err.group(1))  if m_err  else 0,
        "output":  out[-3000:],
    }


def git_restore(paths: list[Path]):
    for p in paths:
        subprocess.run(
            ["git", "checkout", "--", str(p.relative_to(CASASOL))],
            cwd=CASASOL, capture_output=True,
        )


# ─── smart tier: produce (or fall back to) the problem tree ───────────────────

def get_tree(log: dict) -> dict:
    print("\n" + "="*60)
    print("SMART TIER — problem decomposition")
    print("="*60)

    if FORCE_FALLBACK:
        print("  FORCE_FALLBACK=True — using pre-designed fallback tree")
        log["tree_source"] = "fallback"
        return FALLBACK_TREE

    mcp_head = MCP_FILE.read_text()[:3000]  # module header + key imports + first few functions

    messages = [{"role": "user", "content": f"""\
You are a planning agent. Decompose the following task into a tree of leaf tasks.
Each leaf must be a single-function, single-file transduction: one code snippet in → one function out.
Output ONLY a JSON code fence. No prose.

Task:
{TASK_DESC}

Relevant code (first 3000 chars of scripts/mcp_server.py):
```python
{mcp_head}
```

Output schema (no other keys):
{{
  "task": "<one-line summary>",
  "leaves": [
    {{
      "id": "<leaf-A|B|C|D>",
      "deps": ["<leaf-id>", ...],
      "file": "<relative path from casasol/>",
      "target_symbol": "<function name to produce or modify>",
      "instruction": "<precise single-sentence instruction for the cheap model>"
    }}
  ]
}}

Rules:
- Each leaf must target exactly one symbol (function to write or replace).
- deps lists leaf ids whose output must be injected as context before this leaf runs.
- instruction must NOT mention the full task — only the one-function change.
"""}]

    r = call_model(SMART_URL, SMART_MODEL, messages, "smart/plan", timeout=90)
    log["smart_plan_call"] = r

    if not r["ok"]:
        print("  Smart tier failed — using fallback tree")
        log["tree_source"] = "fallback"
        return FALLBACK_TREE

    # Extract JSON fence
    m = re.search(r"```json\s*\n(.*?)```", r["content"], re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"\{.*\"leaves\".*\}", r["content"], re.DOTALL)
    if not m:
        print("  Smart tier produced no valid JSON — using fallback tree")
        log["tree_source"] = "fallback"
        return FALLBACK_TREE

    # m is either a fence match (group 1 = content) or a bare-JSON match (group 0 = content)
    json_str = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
    try:
        tree = json.loads(json_str)
    except Exception as exc:
        print(f"  JSON parse failed ({exc}) — using fallback tree")
        log["tree_source"] = "fallback"
        return FALLBACK_TREE

    if "leaves" not in tree or not tree["leaves"]:
        print("  Tree has no leaves — using fallback tree")
        log["tree_source"] = "fallback"
        return FALLBACK_TREE

    print(f"  Tree produced: {len(tree['leaves'])} leaves")
    log["tree_source"] = "smart"
    return tree


# ─── orchestrator: traverse tree, cheap tier executes leaves ──────────────────

def execute_tree(tree: dict, log: dict) -> tuple[int, dict]:
    """
    Traverse the leaf list in dependency order.
    Orchestrator injects outputs of completed deps as code snippets (not prose).
    Returns (total_interventions, outputs_by_leaf_id).
    """
    print("\n" + "="*60)
    print("LEAF EXECUTOR — deterministic traversal")
    print("="*60)

    leaves     = {l["id"]: l for l in tree["leaves"]}
    outputs    = {}   # leaf_id → code string (the artifact, not prose)
    leaf_logs  = []
    total_iv   = 0

    # Topological order: leaves with no unresolved deps first
    pending = list(tree["leaves"])
    done    = set()
    ordered = []
    MAX_ITER = len(pending) * 2
    i = 0
    while pending and i < MAX_ITER:
        i += 1
        for leaf in pending:
            if all(d in done for d in leaf.get("deps", [])):
                ordered.append(leaf)
                pending.remove(leaf)
                done.add(leaf["id"])
                break

    if pending:
        print(f"  WARNING: unresolvable deps for {[l['id'] for l in pending]} — appending anyway")
        ordered.extend(pending)

    for leaf in ordered:
        lid         = leaf["id"]
        instruction = leaf["instruction"]
        symbol      = leaf["target_symbol"]
        file_rel    = leaf["file"]
        deps        = leaf.get("deps", [])

        file_path = CASASOL / file_rel

        print(f"\n  [{lid}]  target={symbol!r}  file={file_rel}  deps={deps}")

        # ── Gather context snippets (deterministic code join) ───────────────
        snippets = []
        # 1. Current target function (if exists)
        current = read_function(file_path, symbol)
        if current:
            snippets.append(current)
        else:
            # For new functions, show the insertion point
            anchor = "format_listing" if "format_listing" in symbol else None
            if anchor:
                anchor_src = read_function(file_path, anchor)
                snippets.append(f"# Insert before this function:\n{anchor_src[:400]}")

        # 2. Inject dep artifacts (code fence, not prose) — THIS IS THE GLUE JOIN
        for dep_id in deps:
            if dep_id in outputs:
                snippets.append(f"# Artifact from {dep_id} (already applied):\n{outputs[dep_id]}")

        # 3. For test leaf: include fixture boilerplate
        if file_rel.startswith("tests/"):
            header = "\n".join(TEST_FILE.read_text().splitlines()[:55])
            snippets.insert(0, f"# Test file header (imports + fixtures):\n{header}")

        # ── Call cheap tier (strict leaf contract) ──────────────────────────
        result = leaf_call(lid, instruction, snippets, symbol)
        total_iv += result["interventions"]

        # ── Apply output (deterministic code) ───────────────────────────────
        applied = False
        if result["code"]:
            code = result["code"]

            if file_rel.startswith("tests/"):
                applied = append_tests(file_path, code)
                if applied:
                    print(f"  [{lid}] Appended {symbol!r} to {file_rel}")
            elif current:
                applied = apply_replacement(file_path, symbol, code)
                if applied:
                    print(f"  [{lid}] Replaced {symbol!r} in {file_rel}")
            else:
                applied = insert_before(file_path, "format_listing", code)
                if applied:
                    print(f"  [{lid}] Inserted {symbol!r} before format_listing in {file_rel}")

        if not applied:
            print(f"  [{lid}] APPLY FAILED — intervention")
            total_iv += 1
        else:
            # Store artifact for downstream deps (code, not prose)
            outputs[lid] = result["code"]

        leaf_logs.append({
            "leaf":         leaf,
            "calls":        result["calls"],
            "interventions": result["interventions"],
            "applied":      applied,
            "artifact_len": len(result["code"]) if result["code"] else 0,
        })

    log["leaf_logs"] = leaf_logs
    return total_iv, outputs


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = {"experiment": "exp_016b", "run_ts": run_ts, "task": TASK_DESC}

    print(f"\nExp 016B — Local LLM as leaf executor")
    print(f"Started: {run_ts}")

    pre_test = run_tests()
    log["pre_test"] = pre_test
    print(f"\nBaseline: {pre_test['passed']} passed, {pre_test['failed']} failed")

    # ── 1. Smart tier: decompose task into leaf tree ───────────────────────
    tree = get_tree(log)
    log["tree"] = tree
    OUT_TREE.write_text(json.dumps(tree, indent=2))
    print(f"  Tree written: {OUT_TREE} ({len(tree['leaves'])} leaves)")

    # ── 2. Orchestrator + cheap tier: execute leaves ───────────────────────
    total_interventions, outputs = execute_tree(tree, log)

    # ── 3. Run tests ───────────────────────────────────────────────────────
    post_test = run_tests()
    log["post_test"] = post_test

    delta_passed = post_test["passed"] - pre_test["passed"]
    test_ok = post_test["returncode"] == 0

    print(f"\n{'='*60}")
    print("  EXP 016B RESULTS")
    print(f"{'='*60}")
    print(f"  Tree source:       {log.get('tree_source', 'unknown')}")
    print(f"  Leaves executed:   {len(tree['leaves'])}")
    print(f"  Interventions:     {total_interventions}")
    print(f"  Tests before:      {pre_test['passed']}")
    print(f"  Tests after:       {post_test['passed']}  (+{delta_passed} new)")
    print(f"  Suite:             {'PASS' if test_ok else 'FAIL'} ({post_test['failed']} failed)")
    print(f"  Written:           {OUT_LOG}")

    log["summary"] = {
        "tree_source":       log.get("tree_source", "unknown"),
        "leaves":            len(tree["leaves"]),
        "total_interventions": total_interventions,
        "tests_before":      pre_test["passed"],
        "tests_after":       post_test["passed"],
        "new_tests":         delta_passed,
        "suite_pass":        test_ok,
    }
    OUT_LOG.write_text(json.dumps(log, indent=2))

    if not test_ok:
        print("\n  Test output (tail):")
        print(post_test["output"][-1500:])
        print("\n  Restoring files to git HEAD.")
        git_restore([MCP_FILE, TEST_FILE])
        sys.exit(1)


if __name__ == "__main__":
    main()
