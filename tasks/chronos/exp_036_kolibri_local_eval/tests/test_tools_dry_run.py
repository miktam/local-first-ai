"""tools/dry_run.py helpers that need no model (review fix 2026-10-03): a copy of a kit that is already in its run
is reset to its pre-registration state before the dry run fills, scales and builds its own."""

from __future__ import annotations

import re
from pathlib import Path

from tools import dry_run, hash_tree as ht

EXP = Path(__file__).resolve().parents[1]


def test_reset_run_state_unfills_and_removes_real_manifests_and_shingles(tmp_path):
    exp = tmp_path / "exp"
    (exp / "tasks" / "manifests").mkdir(parents=True)
    (exp / "tools").mkdir()
    real = ht.unfill((EXP / "HYPOTHESIS.md").read_text(encoding="utf-8"))
    filled = real
    for i, s in enumerate(ht.SCOPES):                                   # a filled, stamped, amended copy
        filled = filled.replace("{{" + s.placeholder + "}}", f"{i:064x}", 1)
    filled = filled.replace("{{PREREG_UTC}}", "2026-10-04T06:00:00Z")
    filled += "\n## Amendment 0 — sign-off choices (2026-10-04T07:00:00Z)\n\nhash_tree: ANALYSIS_SHA256 = " + "e" * 64 + "\n"
    (exp / "HYPOTHESIS.md").write_text(filled, encoding="utf-8")
    for name in ("gpqa_diamond_en.json", "mmlu_prox_category_counts.json", "ifbench.json", "ifbench_pilot.json"):
        (exp / "tasks" / "manifests" / name).write_text("{}\n")
    (exp / "tools" / "withheld_shingles.sha256").write_text("# x\n")
    done = dry_run.reset_run_state(exp)
    text = (exp / "HYPOTHESIS.md").read_text(encoding="utf-8")
    assert {s.placeholder for s in ht.SCOPES} <= set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text))
    assert "{{PREREG_UTC}}" in text and not ht.AMEND_LINE.search(text)
    assert sorted(p.name for p in (exp / "tasks" / "manifests").iterdir()) == ["ifbench.json", "ifbench_pilot.json"]
    assert not (exp / "tools" / "withheld_shingles.sha256").exists()
    assert "tools/withheld_shingles.sha256" in done
    assert dry_run.reset_run_state(exp) == []                           # idempotent
