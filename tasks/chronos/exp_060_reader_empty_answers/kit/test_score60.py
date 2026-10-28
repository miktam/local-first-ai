#!/usr/bin/env python3
"""Unit test for score60.py on a synthesized run built from exp_055's real manifest and G baseline. No model is called.

    python3.14 -I kit/test_score60.py
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXP55 = HERE.parent / "exp_055_exposure_review_lures"
MAN = json.loads((EXP55 / "data/MANIFEST.json").read_text())
G = {json.loads(l)["id"]: json.loads(l) for l in (EXP55 / "runs/20261025T095614Z/G/results.jsonl").read_text().splitlines() if l}
EMPTY = sorted(i for i, r in G.items() if r["band"] is None)


def right_band(m):
    return "suspicious" if m["kind"] == "lure" else "benign"


def items(m):
    return [x for x in m["used_items"] if not x.startswith("C-")]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def audit_lines(bad=False):
    lines = [{"kind": "getaddrinfo", "host": "127.0.0.1", "port": 11434, "addrs": ["127.0.0.1"]},
             {"kind": "audit", "event": "socket.connect", "args": ["<socket>", "('127.0.0.1', 11434)"]},
             {"kind": "audit", "event": "subprocess.Popen", "args": ["['ollama', 'ps']", "None", "None"]}]
    if bad:
        lines.append({"kind": "audit", "event": "socket.connect", "args": ["<socket>", "('198.51.100.7', 443)"]})
    return lines


def daemon(ctx, vram=18e9):
    return {"api_ps": {"models": [{"name": "gemma4:26b", "context_length": ctx, "size_vram": vram}]}}


def build(run: Path, *, t_empties=0, c_ctx=32768, bad_audit=False, c_flips=0):
    # F pass 1 = G, with counters; empties at the wall
    pass1, retry, final = [], [], []
    for i in sorted(MAN):
        g = G[i]
        if g["band"] is None:
            p = {"id": i, "pass": "pass1", "band": None, "used_items": [], "why": None, "parsed": False, "error": None,
                 "done_reason": "length", "prompt_eval_count": 900, "eval_count": 7292, "seconds": 150.0}
            r = {"id": i, "pass": "retry", "band": right_band(MAN[i]), "used_items": items(MAN[i]), "why": "x",
                 "parsed": True, "error": None, "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 60,
                 "seconds": 4.0}
            pass1.append(p); retry.append(r)
            final.append({**r, "fallback": True, "seconds": 154.0})
        else:
            p = {"id": i, "pass": "pass1", "band": g["band"], "used_items": g["used_items"], "why": g["why"],
                 "parsed": True, "error": None, "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 1500,
                 "seconds": g["seconds"]}
            pass1.append(p); final.append({**p, "fallback": False})
    write(run / "F/pass1.jsonl", pass1); write(run / "F/retry.jsonl", retry); write(run / "F/results.jsonl", final)
    # T: every message answered right, fast; optionally some empties
    t = []
    for n, i in enumerate(sorted(MAN)):
        if n < t_empties:
            t.append({"id": i, "band": None, "used_items": [], "parsed": False, "error": None, "done_reason": "stop",
                      "prompt_eval_count": 900, "eval_count": 5, "seconds": 1.0})
        else:
            t.append({"id": i, "band": right_band(MAN[i]), "used_items": items(MAN[i]), "parsed": True, "error": None,
                      "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 60, "seconds": 4.0})
    write(run / "T/results.jsonl", t)
    # C: G's answers kept (optionally some flipped), empties recovered right after long thinking
    c, flipped = [], 0
    for i in sorted(MAN):
        g = G[i]
        if g["band"] is None:
            c.append({"id": i, "band": right_band(MAN[i]), "used_items": items(MAN[i]), "parsed": True, "error": None,
                      "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 9000, "seconds": 120.0})
        else:
            band = g["band"]
            if flipped < c_flips and i not in {"X011", "X067", "X085", "X114", "X148", "X161", "X174"}:
                band = "benign" if band != "benign" else "suspicious"; flipped += 1
            c.append({"id": i, "band": band, "used_items": g["used_items"], "parsed": True, "error": None,
                      "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 1500, "seconds": g["seconds"]})
    write(run / "C/results.jsonl", c)
    for a in ("T", "F", "C"):
        write(run / a / "audit.jsonl", audit_lines(bad=bad_audit and a == "C"))
    (run / "C/daemon_before.json").write_text(json.dumps(daemon(8192)))
    (run / "C/daemon_during.json").write_text(json.dumps(daemon(c_ctx, 22e9)))
    (run / "C/daemon_restored.json").write_text(json.dumps(daemon(8192)))


def score(run: Path) -> dict:
    p = subprocess.run([sys.executable, str(HERE / "kit/score60.py"), str(run)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return json.loads((run / "score.json").read_text())


class Score60(unittest.TestCase):
    def test_clean_run(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            build(run)
            s = score(run)
            c = s["controls"]
            self.assertTrue(c["C1_baseline_reproduces"]["pass"])
            self.assertEqual(c["C1_baseline_reproduces"]["agreement"]["rate"], 1.0)
            self.assertEqual(c["C1_baseline_reproduces"]["unchanged_empties_empty_again"], 20)
            self.assertTrue(c["C2_wall_is_the_cause"]["pass"])
            self.assertEqual(c["C2_wall_is_the_cause"]["empty_pass1"], 22)
            self.assertTrue(c["C3_loopback_only"]["pass"])
            self.assertTrue(c["C4_arm_C_at_32K"]["pass"])
            for a in ("T", "F", "C"):
                self.assertEqual(s["arms"][a]["answered"], 220)
                self.assertTrue(s["hypotheses"][a]["H1_loss_closed"]["pass"])
                self.assertTrue(s["hypotheses"][a]["H3_no_worse_than_G"]["pass"])
                self.assertTrue(s["hypotheses"][a]["H4_affordable"]["pass"])
            # all 22 recovered right: BA 0.932 for F and C, 1.0 for T
            self.assertAlmostEqual(s["arms"]["F"]["balanced_accuracy"], 0.932, places=3)
            self.assertAlmostEqual(s["arms"]["C"]["balanced_accuracy"], 0.932, places=3)
            self.assertEqual(s["arms"]["T"]["balanced_accuracy"], 1.0)
            self.assertTrue(s["hypotheses"]["C"]["H2_moves_nothing_it_did_not_have_to"]["pass"])
            self.assertEqual(s["arms"]["F"]["fallbacks"], 22)
            self.assertEqual(s["decision"]["chosen"], "T")
            self.assertIs(s["decision"]["thinking_earned_its_place"], False)
            self.assertEqual(s["descriptive"]["recovered_summary"]["F"]["lures_caught"], 11)
            self.assertEqual(s["descriptive"]["budget_curve_from_F_pass1"]["within_2000"]["share"], 1.0)
            self.assertEqual(s["descriptive"]["memory_gemma4_26b"]["during_C"]["context"], 32768)

    def test_failures_are_caught(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            build(run, t_empties=5, c_ctx=8192, bad_audit=True, c_flips=15)
            s = score(run)
            self.assertFalse(s["hypotheses"]["T"]["H1_loss_closed"]["pass"])        # 215 < 218
            self.assertFalse(s["controls"]["C4_arm_C_at_32K"]["pass"])                # daemon ignored num_ctx
            self.assertFalse(s["controls"]["C3_loopback_only"]["pass"])               # an external connect
            self.assertIn("connect", s["controls"]["C3_loopback_only"]["per_arm"]["C"]["violations"][0])
            self.assertFalse(s["hypotheses"]["C"]["H2_moves_nothing_it_did_not_have_to"]["pass"])  # 15/193 flipped
            self.assertNotIn("C", s["decision"]["eligible"])
            self.assertNotIn("T", s["decision"]["eligible"])
            self.assertEqual(s["decision"]["chosen"], "F")

    def test_tie_goes_to_the_faster(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            build(run)
            # make T's accuracy equal to F's (within 0.01) by flipping 15 of T's genuine answers to flagged
            rows = [json.loads(l) for l in open(run / "T/results.jsonl")]
            n = 0
            for r in rows:
                if MAN[r["id"]]["kind"] == "benign" and n < 15:
                    r["band"] = "suspicious"; n += 1
            with open(run / "T/results.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            s = score(run)
            self.assertAlmostEqual(s["arms"]["T"]["balanced_accuracy"], 1 - 15 / 220, places=3)   # 0.932
            self.assertEqual(s["decision"]["chosen"], "T")       # same BA as F and C, T is the fastest
            self.assertIn("lower median wall-clock", s["decision"]["why"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
