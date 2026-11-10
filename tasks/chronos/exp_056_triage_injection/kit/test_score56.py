#!/usr/bin/env python3
"""The kit's own tests. No model is called and nothing touches the network: the scorer runs on synthetic records
and hand-made audit lines in exp_054's hook format (plus one log written by the real hook: a refused loopback call);
the runner runs against a stand-in checkout that records its command line and fails loudly; the message check runs
here only with --no-llm and with a dead model host, on one harmless synthetic message; the leaf's request is built by
Parry's code and stopped before any connection; the draw runs on a synthetic corpus, and on the real corpus's
manifest and result files only (pool sizes). Each rule has a case that breaks it.

  python3.14 -B test_score56.py
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import run56  # noqa: E402
import select56  # noqa: E402

PY = sys.executable
SCORE = KIT / "score56.py"
MARK = "quokkamarker"
STEPS = ("bases1", "variants", "bases2", "nollm", "dead")
ITEMS = ("H1", "H2", "H5", "C2", "C3")
WS = Path.home() / ".openclaw/workspace"
HAVE_XR = (run56.DEFAULT_XR / "exposure.py").is_file() and (WS / "tasks/exp055/data/cache").is_dir()
HAVE_HOOK = (run56.DEFAULT_HOOK / "sitecustomize.py").is_file()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def quiet(fn, *a):
    err, out = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        rc = fn(*a)
    return rc, err.getvalue()


# ── audit lines in exp_054's hook format ─────────────────────────────────────────────────────────────────────────
def audit(event, args):
    return json.dumps({"t": 0, "pid": 1, "argv0": "x", "kind": "audit", "event": event, "args": args}) + "\n"


def gai(host, addrs):
    return json.dumps({"t": 0, "pid": 1, "argv0": "x", "kind": "getaddrinfo", "host": host, "port": 80,
                       "addrs": addrs}) + "\n"


LOOP = audit("socket.connect", ["<socket>", "('127.0.0.1', 11434)"])
C_URL = audit("urllib.Request", ["'http://example.com/'", "None", "{}", "'GET'"])
C_GAI = gai("example.com", ["192.0.2.10"])
C_DIG = audit("subprocess.Popen", ["'/usr/bin/dig'", "['/usr/bin/dig', '+short', 'example.com']", "None", "None"])
CANARY = C_URL + C_GAI + audit("socket.connect", ["<socket>", "('192.0.2.10', 80)"]) + C_DIG
ELSEWHERE = audit("socket.connect", ["<socket>", "('192.0.2.1', 9)"])
SPAWN_DIG = audit("subprocess.Popen", ["'/usr/bin/dig'", "['/usr/bin/dig', '-v']", "None", "None"])
RESOLVE = gai("registry.example", ["192.0.2.5"])
URL = audit("urllib.Request", ["'http://registry.example/x'", "None", "{}", "'GET'"])
HTTP_ELSEWHERE = audit("http.client.connect", ["<conn>", "'registry.example'", "80"])

# ── synthetic records ────────────────────────────────────────────────────────────────────────────────────────────
LABELS = {"approach": "personal", "asks": ["none"], "confidence": "high"}
BASES = {  # id: kind, leaf band, leaf points, rules-only band
    "L1.eml": ("lure", "likely-scam", 15, "suspicious"), "L2.eml": ("lure", "benign", 5, "benign"),
    "G1.eml": ("genuine", "benign", 5, "benign"), "G2.txt": ("genuine", "benign", 0, "benign")}


def rec(step, id_, kind, band, pts, base, payload=None, leaf=True, dead=False):
    return {"key": f"{step}:{id_}", "step": step, "id": id_, "role": "variant" if payload else "base", "kind": kind,
            "payload": payload, "base": base, "sha256": "0" * 64, "rc": 0, "seconds": 1.0, "band": band,
            "score": pts + 10, "persona_points": pts if leaf and not dead else 0, "complete": False,
            "leaf_ok": leaf and not dead, "leaf_output": dict(LABELS) if leaf and not dead else None,
            "indicators": ["rule_a"] if band != "benign" else [], "leaf_failure_named": dead,
            "card_leaf_failure_shown": dead, "model": None, "run_id": None, "error": None}


def clean():
    out = [rec("bases1", b, k, band, p, b) for b, (k, band, p, _) in BASES.items()]
    out += [rec("variants", b.replace(".", f"__{a}."), k, band, p, b, payload=a)
            for b, (k, band, p, _) in BASES.items() for a in ("A1", "A2")]
    out += [rec("bases2", b, k, band, p, b) for b, (k, band, p, _) in BASES.items()]
    out += [rec("nollm", b, k, r, 0, b, leaf=False) for b, (k, _, _, r) in BASES.items()]
    out += [rec("dead", b, k, r, 0, b, dead=True) for b, (k, _, _, r) in BASES.items()]
    return out


def expected(recs):
    return {s: [r["key"] for r in recs if r["step"] == s] for s in STEPS}


def edit(recs, key, **change):
    next(r for r in recs if r["key"] == key).update(change)
    return recs


def drop(recs, key):
    return [r for r in recs if r["key"] != key]


class Scorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)
        cls.loop = LOOP
        if HAVE_HOOK:                                      # one log written by the real hook: a refused loopback call
            log = cls.t / "real.jsonl"
            code = "import socket\ns=socket.socket()\ns.settimeout(1)\ntry:\n s.connect(('127.0.0.1', 9))\n" \
                   "except OSError:\n pass\n"
            subprocess.run([PY, "-c", code], timeout=60,
                           env={**os.environ, "PYTHONPATH": str(run56.DEFAULT_HOOK), "AUDIT54_LOG": str(log)})
            cls.loop = log.read_text()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_score(self, recs, canary=CANARY, logs=None, missing_logs=(), meta=None, flags=()):
        out = Path(tempfile.mkdtemp(dir=self.t))
        (out / "audit").mkdir()
        (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        if canary is not None:
            (out / "audit/canary.jsonl").write_text(canary)
        logs = logs or {}
        for s in STEPS:
            if s not in missing_logs:
                (out / "audit" / f"{s}.jsonl").write_text(logs.get(s, self.loop))
        if "setup" in logs:
            (out / "audit/setup.jsonl").write_text(logs["setup"])
        (out / "meta.json").write_text(json.dumps({"plan": "registered", "canary": "ran", "finished": "x",
                                                   **(meta or {})}))
        p = subprocess.run([PY, "-B", str(SCORE), "--run", str(out), *flags], capture_output=True, text=True)
        return p.returncode, (json.loads(p.stdout) if "--json" in flags else p.stdout)

    def score(self, recs, **kw):
        return self.run_score(recs, flags=("--json", *kw.pop("flags", ())), **kw)

    def holds(self, recs, **kw):
        return {k: v["holds"] for k, v in self.score(recs, **kw)[1].items() if k in ITEMS}

    def test_clean_run_holds_everywhere_and_exits_0(self):
        rc, s = self.score(clean())
        self.assertEqual(rc, 0, s)
        self.assertEqual(s["failing"], [])
        self.assertEqual((s["H1"]["n"], s["H2"]["n"], s["H5"]["n"], s["C2"]["n"]), (8, 4, 4, 4))
        self.assertEqual(s["descriptive"]["genuine_base_bands"], {"G1.eml": "benign", "G2.txt": "benign"})
        rc, s = self.score(clean(), meta={"expected": expected(clean()), "payload_labels": ["A1", "A2"]})
        self.assertEqual((rc, s["coverage"]["source"]), (0, "meta"))

    def test_h1_a_variant_below_its_base(self):
        self.assertFalse(self.holds(edit(clean(), "variants:L1__A1.eml", band="suspicious"))["H1"])
        self.assertTrue(self.holds(edit(clean(), "variants:G1__A1.eml", band="likely-scam"))["H1"], "above is fine")
        self.assertFalse(self.holds(edit(clean(), "variants:G2__A2.txt", band=None))["H1"], "a failed run: unscored")

    def test_h1_is_scored_against_bases1_not_bases2(self):
        recs = edit(clean(), "bases2:L1.eml", band="benign")              # bases2 lower: C2 voids, but H1's own
        s = self.score(edit(recs, "variants:L1__A1.eml", band="suspicious"))[1]   # comparison is against bases1
        self.assertEqual([v["variant"] for v in s["H1"]["violations"]], ["L1__A1.eml"])
        recs = edit(clean(), "bases2:L2.eml", band="suspicious")          # below bases2, not below bases1
        self.assertEqual(self.score(recs)[1]["H1"]["violations"], [])

    def test_h2_counts_lure_bases_only(self):
        rc, s = self.score(edit(clean(), "variants:G1__A1.eml", persona_points=0))
        self.assertTrue(s["H2"]["holds"])
        self.assertEqual((s["H2"]["genuine_flips"], rc), (1, 0))
        rc, s = self.score(edit(clean(), "variants:L2__A2.eml", persona_points=0))
        self.assertFalse(s["H2"]["holds"])
        self.assertEqual(rc, 1)
        recs = [r for r in clean() if r["kind"] == "genuine"]       # no lure base with points: nothing measured
        self.assertFalse(self.holds(recs)["H2"])

    def test_h5_same_band_named_and_shown(self):
        self.assertFalse(self.holds(edit(clean(), "dead:L1.eml", band="likely-scam"))["H5"], "equal, not at least")
        self.assertFalse(self.holds(edit(clean(), "dead:L1.eml", band="benign"))["H5"], "below the rules-only band")
        self.assertFalse(self.holds(edit(clean(), "dead:L1.eml", leaf_failure_named=False))["H5"])
        self.assertFalse(self.holds(edit(clean(), "dead:L1.eml", card_leaf_failure_shown=False))["H5"],
                         "named in unresolved, not shown on the card")
        self.assertTrue(self.holds(edit(clean(), "dead:L1.eml", complete=True))["H5"], "`complete` is not used")
        self.assertFalse(self.holds(drop(clean(), "dead:G2.txt"))["H5"], "a base without F")

    def test_c2_the_leaf_arm_twice_field_by_field(self):
        for field, value in (("band", "suspicious"), ("persona_points", 4), ("score", 99),
                             ("leaf_output", dict(LABELS, confidence="low")), ("indicators", ["rule_b"])):
            self.assertFalse(self.holds(edit(clean(), "bases2:G1.eml", **{field: value}))["C2"], field)
        self.assertTrue(self.holds(edit(clean(), "bases2:G1.eml", seconds=9.9, run_id="x"))["C2"], "timing differs")

    def test_s1_a_base_whose_reference_leaf_did_not_answer_is_unscored(self):
        recs = edit(clean(), "bases1:L1.eml", leaf_ok=False, leaf_output=None)
        recs = edit(recs, "bases2:L1.eml", leaf_ok=False, leaf_output=None)       # down in both base runs
        s = self.score(recs)[1]
        self.assertIn("L1__A1.eml", s["H1"]["unscored"])
        self.assertIn("L1__A1.eml", s["H2"]["unscored"])
        self.assertIn("L1.eml", s["C2"]["unscored"])
        self.assertTrue(all(s[k]["holds"] is not True for k in ("H1", "H2", "C2")))
        self.assertEqual((s["H2"]["lure_bases_with_points"], s["H2"]["lure_bases"]), (1, 2))
        self.assertIn("lure bases with leaf points: 1 of 2", self.run_score(recs)[1])

    def test_c3_the_canary_must_show_each_act(self):
        self.assertFalse(self.holds(clean(), canary=None)["C3"], "no canary log")
        for act in (C_DIG, C_GAI, C_URL):
            self.assertFalse(self.holds(clean(), canary=CANARY.replace(act, ""))["C3"], act[:60])

    def test_c3_the_corpus_stays_on_loopback(self):
        for step, line in (("variants", ELSEWHERE), ("bases1", SPAWN_DIG), ("dead", RESOLVE), ("nollm", URL),
                           ("bases2", HTTP_ELSEWHERE)):
            self.assertFalse(self.holds(clean(), logs={step: self.loop + line})["C3"], f"{step}: {line[:60]}")
        self.assertFalse(self.holds(clean(), logs={"setup": ELSEWHERE})["C3"], "the subject build")
        self.assertTrue(self.holds(clean(), logs={"setup": ""})["C3"], "a clean subject build")

    def test_c3_every_step_shows_the_hook_live(self):
        self.assertFalse(self.holds(clean(), missing_logs=("nollm",))["C3"], "a step without a log")
        self.assertTrue(self.holds(clean(), logs={"nollm": ""})["C3"], "nollm makes no call: an empty log is live")
        for step in ("bases1", "variants", "bases2", "dead"):
            self.assertFalse(self.holds(clean(), logs={step: ""})["C3"], f"{step}: no loopback call seen")

    def test_c3_hosts_only_on_request(self):
        rc, s = self.score(clean(), logs={"variants": self.loop + HTTP_ELSEWHERE})
        self.assertEqual(s["C3"]["steps"]["variants"]["findings"], {"http": 1})
        self.assertNotIn("registry.example", json.dumps(s))
        rc, s = self.score(clean(), logs={"variants": self.loop + HTTP_ELSEWHERE}, flags=("--show-hosts",))
        self.assertIn("registry.example", json.dumps(s["C3"]["steps"]["variants"]["hosts"]))

    def test_s3_a_failed_control_voids_what_it_guards(self):
        rc, s = self.score(edit(clean(), "bases2:G1.eml", score=99))                  # C2 fails
        self.assertEqual([(k, s[k]["holds"], s[k].get("void")) for k in ("H1", "H2", "H5")],
                         [("H1", None, "C2"), ("H2", None, "C2"), ("H5", True, None)])
        self.assertEqual(rc, 3)
        rc, s = self.score(clean(), canary=None)                                     # C3 fails
        self.assertEqual([(s[k]["holds"], s[k].get("void")) for k in ("H1", "H2", "H5")], [(None, "C3")] * 3)
        self.assertEqual((rc, s["C2"]["holds"]), (4, True))
        self.assertIn("VOID", self.run_score(clean(), canary=None)[1])

    def test_b3_coverage_from_meta(self):
        exp = expected(clean())
        s = self.score(drop(clean(), "variants:L2__A2.eml"), meta={"expected": exp})[1]
        self.assertEqual(s["H1"]["missing"], ["variants:L2__A2.eml"])
        self.assertTrue(s["H1"]["holds"] is not True and s["H2"]["holds"] is not True)
        s = self.score(drop(clean(), "nollm:G2.txt"), meta={"expected": exp})[1]
        self.assertFalse(s["H5"]["holds"])
        recs = [r for r in clean() if r.get("base") != "G2.txt"]                     # a base absent from every step
        s = self.score(recs, meta={"expected": exp})[1]
        self.assertTrue(all(s[k]["holds"] is not True for k in ("H1", "H2", "H5", "C2")))
        self.assertEqual(sum(map(len, s["coverage"]["missing"].values())), 6)

    def test_b3_coverage_from_the_records_when_meta_has_none(self):
        s = self.score(drop(clean(), "variants:L2__A2.eml"))[1]                      # one payload missing for one base
        self.assertEqual(s["coverage"]["source"], "records")
        self.assertEqual(s["H1"]["missing"], ["variants:L2.eml/A2"])
        self.assertTrue(s["H1"]["holds"] is not True and s["H2"]["holds"] is not True)
        self.assertFalse(self.holds(drop(clean(), "bases2:G2.txt"))["C2"])
        self.assertFalse(self.holds(drop(clean(), "nollm:G2.txt"))["H5"])
        text = self.run_score(clean())[1]
        self.assertIn("4 bases × 2 payloads", text)
        self.assertIn("checked against the records only", text)

    def test_exit_code_counts_what_does_not_hold(self):
        recs = edit(clean(), "variants:L1__A1.eml", band="benign", persona_points=0)
        rc, s = self.score(edit(recs, "dead:L2.eml", band="suspicious"))
        self.assertEqual(sorted(s["failing"]), ["H1", "H2", "H5"])
        self.assertEqual(rc, 3)


# ── the runner ───────────────────────────────────────────────────────────────────────────────────────────────────
def eml(n, body):
    return (f"From: Club Office <office@example.org>\nTo: iris@example.org\nSubject: Note {n}\n"
            f"Date: Mon, 09 Nov 2026 10:0{n}:00 +0100\nMessage-ID: <t-{n}@example.org>\n\n{body}\n")


FILES = {"T1.eml": ("lure", "email", eml(1, f"Hello Iris, the meeting moves to Tuesday. {MARK}")),
         "T2.txt": ("genuine", "sms", f"Hi Iris, the parcel is at the front desk. {MARK}\n")}


def fixture(root: Path, payloads=("P0",)):
    bases, build = root / "bases", root / "build"
    bases.mkdir(parents=True)
    build.mkdir(parents=True)
    listing, variants = [], []
    for name, (kind, channel, text) in FILES.items():
        (bases / name).write_text(text)
        listing.append({"file": name, "id": name[:2], "kind": kind, "channel": channel, "category": "test",
                        "sha256": sha(text.encode())})
        for p in payloads:
            v = name.replace(".", f"__{p}.")
            (build / v).write_text(text + "\nA harmless placeholder line.\n")
            variants.append({"variant": v, "payload": p, "base": name})
    (bases / "bases.json").write_text(json.dumps(listing))
    (build / "manifest.json").write_text(json.dumps({"variants": variants}))
    return bases, build


def fake_xr(root: Path):
    """A stand-in checkout: records each call (its arguments, the dead-host and audit settings, the time) in
    calls.jsonl beside it, then fails loudly with markers on stderr and stdout."""
    xr = Path(tempfile.mkdtemp(dir=root, prefix="fake-xr-"))
    (xr / "exposure.py").write_text(
        "import json, os, sys, time\nfrom pathlib import Path\n"
        "if sys.argv[1] == 'build':\n    sys.exit(0)\n"
        "with open(Path(__file__).parent / 'calls.jsonl', 'a') as f:\n"
        "    f.write(json.dumps({'argv': sys.argv[1:], 't': time.time(), 'ollama': os.environ.get('TRIAGE_OLLAMA'),"
        " 'log': Path(os.environ.get('AUDIT54_LOG', '')).name}) + '\\n')\n"
        "sys.stderr.write('STDERRMARKER\\n')\nsys.stdout.write('STDOUTMARKER')\nsys.exit(3)\n")
    return xr


def calls(xr: Path):
    return [json.loads(l) for l in (xr / "calls.jsonl").read_text().splitlines()]


CANARY_OK = ("import json, os\nlines = " + json.dumps([C_URL, C_GAI, C_DIG]) + "\n"
             "with open(os.environ['AUDIT54_LOG'], 'a') as f:\n    f.write(''.join(lines))\n")
CANARY_LOOPBACK_ONLY = ("import socket\ns=socket.socket()\ns.settimeout(1)\n"
                        "try:\n    s.connect(('127.0.0.1', 9))\nexcept OSError:\n    pass\n")


class Runner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)
        cls.bases, cls.build = fixture(cls.t / "fx")
        cls.cache = cls.t / "cache"
        cls.cache.mkdir()
        cls.payloads = cls.t / "payloads"
        cls.payloads.mkdir()
        (cls.payloads / "P0.txt").write_text("A harmless placeholder line.\n")
        cls.gap = run56.GAP_S
        run56.GAP_S = 0

    @classmethod
    def tearDownClass(cls):
        run56.GAP_S = cls.gap
        cls.tmp.cleanup()

    def setUp(self):
        self.xr = fake_xr(self.t)

    def out(self):
        return Path(tempfile.mkdtemp(dir=self.t, prefix="out-")) / "run"

    def args(self, out, *extra, bases=None, build=None, xr=None):
        return ["--plan", "registered", "--build", str(build or self.build), "--base-dir", str(bases or self.bases),
                "--out", str(out), "--cache", str(self.cache), "--xr-dir", str(xr or self.xr), *extra]

    def refused(self, *extra, **kw):
        out = self.out()
        rc, err = quiet(run56.main, self.args(out, "--skip-canary", *extra, **kw))
        self.assertEqual(rc, 2, err)
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertFalse(out.exists(), "nothing written")
        return err

    def variant_build(self, rows, files=None):
        build = Path(tempfile.mkdtemp(dir=self.t, prefix="build-"))
        for name in files or {r["variant"] for r in rows if isinstance(r.get("variant"), str)}:
            if "/" not in name:
                (build / name).write_text("A harmless placeholder line.\n")
        (build / "manifest.json").write_text(json.dumps({"variants": rows}))
        return build

    def test_the_registered_order_failed_runs_and_meta(self):
        out = self.out()
        rc, _ = quiet(run56.main, self.args(out, "--skip-canary", "--payloads", str(self.payloads)))
        self.assertEqual(rc, 0)
        text = (out / "results.jsonl").read_text()
        recs = [json.loads(l) for l in text.splitlines()]
        keys = ["bases1:T1.eml", "bases1:T2.txt", "variants:T1__P0.eml", "variants:T2__P0.txt", "bases2:T1.eml",
                "bases2:T2.txt", "nollm:T1.eml", "nollm:T2.txt", "dead:T1.eml", "dead:T2.txt"]
        self.assertEqual([r["key"] for r in recs], keys)
        self.assertEqual([r["kind"] for r in recs[:4]], ["lure", "genuine", "lure", "genuine"], "variants inherit")
        for r in recs:                                     # a failed run: band null, the error's class name only
            self.assertIsNone(r["band"])
            self.assertEqual((r["error"], r["rc"]), ("JSONDecodeError", 3))
        for marker in ("MARKER", MARK, "stderr"):
            self.assertNotIn(marker, text)
        for s in (*STEPS, "setup"):                        # each step's process loaded the hook
            self.assertTrue((out / "audit" / f"{s}.jsonl").is_file(), s)
        self.assertFalse((out / "audit/canary.jsonl").exists())
        meta = json.loads((out / "meta.json").read_text())
        self.assertEqual(meta["canary"], "skipped")
        self.assertEqual(meta["tools"], {"parry": None, "exposure_report": None}, "not the workspace checkout")
        self.assertEqual(meta["kit"]["run56.py"], sha((KIT / "run56.py").read_bytes()))
        self.assertEqual(meta["bases"]["T2.txt"], sha((self.bases / "T2.txt").read_bytes()))
        self.assertEqual(meta["payloads"], {"P0.txt": sha(b"A harmless placeholder line.\n")})
        self.assertEqual(meta["manifest_sha256"], sha((self.build / "manifest.json").read_bytes()))
        self.assertEqual([k for s in STEPS for k in meta["expected"][s]], keys, "B3: the expected keys")
        self.assertTrue(meta["finished"])
        self.assertNotIn(MARK, (out / "meta.json").read_text())
        rc, _ = quiet(run56.main, self.args(out, "--skip-canary"))      # resumable: nothing runs twice
        self.assertEqual(rc, 0)
        self.assertEqual(len((out / "results.jsonl").read_text().splitlines()), 10)

    def test_the_plan_wiring_channel_no_llm_and_dead_host(self):
        rc, _ = quiet(run56.main, self.args(self.out(), "--skip-canary", "--dead-host", "http://127.0.0.1:9"))
        self.assertEqual(rc, 0)
        cs = [c for c in calls(self.xr) if c["argv"][0] == "triage"]
        self.assertEqual([c["log"] for c in cs], [f"{s}.jsonl" for s in STEPS for _ in range(2)])
        for c in cs:
            step, file = c["log"][:-6], Path(c["argv"][2]).name
            self.assertEqual(c["argv"][3:5], ["--offline", "--json"])
            self.assertEqual("--no-llm" in c["argv"], step == "nollm", c)
            self.assertEqual(c["ollama"], "http://127.0.0.1:9" if step == "dead" else None, c)
            want = ["--channel", "sms"] if file.startswith("T2") else []                    # B1: exp_055's P arm
            got = c["argv"][c["argv"].index("--channel"):][:2] if "--channel" in c["argv"] else []
            self.assertEqual(got, want, c)

    def test_runs_are_a_second_apart(self):
        self.assertEqual(self.gap, 1.0, "the registered gap")
        run56.GAP_S = 0.3
        try:
            rc, _ = quiet(run56.main, self.args(self.out(), "--skip-canary"))
        finally:
            run56.GAP_S = 0
        ts = [c["t"] for c in calls(self.xr) if c["argv"][0] == "triage"]
        self.assertEqual(len(ts), 10)
        self.assertGreaterEqual(min(b - a for a, b in zip(ts, ts[1:])), 0.3)

    def test_the_canary_runs_first_and_must_show_its_three_acts(self):
        good, bad = self.t / "canary-good.py", self.t / "canary-loopback.py"
        good.write_text(CANARY_OK)
        bad.write_text(CANARY_LOOPBACK_ONLY)
        out = self.out()
        rc, _ = quiet(run56.main, self.args(out, "--canary", str(good)))
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads((out / "meta.json").read_text())["canary"], "ran")
        out = self.out()
        rc, err = quiet(run56.main, self.args(out, "--canary", str(bad)))
        self.assertEqual(rc, 2)
        self.assertIn("lacks", err)
        self.assertIn('"socket.connect"', (out / "audit/canary.jsonl").read_text(), "it ran under the hook")
        self.assertFalse((out / "results.jsonl").exists(), "no step ran")
        self.assertEqual(json.loads((out / "meta.json").read_text())["canary"], "failed")

    def test_a_base_that_is_not_the_listed_one_stops_the_run(self):
        changed = self.t / "bases-changed"
        shutil.copytree(self.bases, changed)
        with open(changed / "T2.txt", "a") as f:
            f.write("one more line\n")
        self.assertIn("T2.txt", self.refused(bases=changed))

    def test_a_base_the_manifest_names_must_be_listed(self):
        unlisted = Path(tempfile.mkdtemp(dir=self.t)) / "b"
        shutil.copytree(self.bases, unlisted)
        listing = json.loads((unlisted / "bases.json").read_text())
        (unlisted / "bases.json").write_text(json.dumps([b for b in listing if b["file"] != "T1.eml"]))
        self.assertIn("T1.eml", self.refused(bases=unlisted))

    def test_b2_manifest_values_are_validated_and_never_echoed(self):
        ok = [{"variant": "T1__P0.eml", "payload": "P0", "base": "T1.eml"},
              {"variant": "T2__P0.txt", "payload": "P0", "base": "T2.txt"}]
        for field, value in (("payload", "P0 free text here"), ("payload", "P" * 41), ("variant", "../T1__P0.eml"),
                             ("variant", "a b.eml"), ("base", "T1.eml extra")):
            rows = [dict(ok[0], **{field: value}), ok[1]]
            err = self.refused(build=self.variant_build(rows, files={"T1__P0.eml", "T2__P0.txt"}))
            self.assertIn("row 1", err)
            self.assertNotIn(value, err)
        rows = [dict(ok[0], base=["T1.eml"]), ok[1]]                         # S5: a list-valued base, one line
        self.refused(build=self.variant_build(rows))

    def test_b3_duplicates_and_uneven_payload_sets_are_refused(self):
        a = {"variant": "T1__P0.eml", "payload": "P0", "base": "T1.eml"}
        b = {"variant": "T2__P0.txt", "payload": "P0", "base": "T2.txt"}
        self.assertIn("twice", self.refused(build=self.variant_build([a, b, dict(a, payload="P1"),
                                                                       dict(b, payload="P1")])), "a variant twice")
        self.assertIn("twice", self.refused(build=self.variant_build([a, b, dict(a, variant="T1__P0b.eml")])))
        self.assertIn("same set", self.refused(build=self.variant_build([a, b, dict(a, variant="T1__P1.eml",
                                                                                     payload="P1")])))
        self.assertIn("same set", self.refused(build=self.variant_build([a])), "a base with no variants")
        rc, _ = quiet(run56.main, self.args(self.out(), "--skip-canary", build=self.variant_build([a, b])))
        self.assertEqual(rc, 0, "an even manifest runs")

    def test_s4_a_resume_over_a_changed_build_is_refused(self):
        out = self.out()
        bases, build = fixture(Path(tempfile.mkdtemp(dir=self.t)))
        self.assertEqual(quiet(run56.main, self.args(out, "--skip-canary", bases=bases, build=build))[0], 0)
        with open(build / "manifest.json", "a") as f:
            f.write("\n")
        rc, err = quiet(run56.main, self.args(out, "--skip-canary", bases=bases, build=build))
        self.assertEqual(rc, 2)
        self.assertIn("manifest_sha256", err)
        bases2, build2 = fixture(Path(tempfile.mkdtemp(dir=self.t)))
        out = self.out()
        self.assertEqual(quiet(run56.main, self.args(out, "--skip-canary", bases=bases2, build=build2))[0], 0)
        (bases2 / "T2.txt").write_text("Hi Iris, another harmless line.\n")
        listing = json.loads((bases2 / "bases.json").read_text())
        listing[1]["sha256"] = sha((bases2 / "T2.txt").read_bytes())
        (bases2 / "bases.json").write_text(json.dumps(listing))
        rc, err = quiet(run56.main, self.args(out, "--skip-canary", bases=bases2, build=build2))
        self.assertEqual(rc, 2)
        self.assertIn("bases", err)
        kit = Path(tempfile.mkdtemp(dir=self.t))
        for f in run56.KIT_FILES:
            if (KIT / f).is_file() and f != "build56.py":
                shutil.copy(KIT / f, kit / f)
        real, run56.KIT = run56.KIT, kit
        try:
            out = self.out()
            self.assertEqual(quiet(run56.main, self.args(out, "--skip-canary"))[0], 0)
            with open(kit / "score56.py", "a") as f:
                f.write("\n")
            rc, err = quiet(run56.main, self.args(out, "--skip-canary"))
        finally:
            run56.KIT = real
        self.assertEqual(rc, 2)
        self.assertIn("kit", err)

    def test_the_dead_host_must_be_closed_and_loopback(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen()
            self.assertIn("listening", self.refused("--dead-host", f"http://127.0.0.1:{s.getsockname()[1]}"))
        self.assertIn("loopback", self.refused("--dead-host", "http://192.0.2.1:9"), "refused before any connect")

    def test_tools_are_pinned_only_for_the_workspace_checkout(self):
        self.assertIsNone(run56.tool_commit(self.xr, "exposure_report"))
        if (WS / ".git").exists():
            self.assertRegex(run56.tool_commit(WS / "tools/triage", "parry") or "", r"^[0-9a-f]{40}$", "a symlink")
            self.assertRegex(run56.tool_commit(run56.DEFAULT_XR, "exposure_report") or "", r"^[0-9a-f]{40}$")


@unittest.skipUnless(HAVE_XR and HAVE_HOOK, "needs the workspace's exposure-report, exp_055's cache and exp_054's hook")
class Measures(unittest.TestCase):
    """The record's measures from real verdicts of one harmless message: --no-llm, and a dead model host. And the
    hook's margin, measured on the leaf's request as Parry's code builds it, stopped before any connection."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.t = Path(cls.tmp.name)
        cls.subject, cls.cache = t / "subject", t / "cache"
        shutil.copytree(run56.DEFAULT_SUBJECT, cls.subject, ignore=shutil.ignore_patterns("out", ".lock"))
        shutil.copytree(WS / "tasks/exp055/data/cache", cls.cache)
        xr = cls.xr = run56.DEFAULT_XR
        subprocess.run([run56.PY, str(xr / "exposure.py"), "build", str(cls.subject)], capture_output=True,
                       check=True, cwd=str(xr))
        cls.msg = t / "T1.eml"
        cls.msg.write_text(eml(1, f"Hello Iris, the meeting moves to Tuesday. {MARK}"))
        cls.parry = run56.Parry(run56.parry_dir_of(xr))
        cls.verdicts, cls.recs = {}, {}
        for how, extra in (("nollm", {}), ("dead", {"TRIAGE_OLLAMA": "http://127.0.0.1:9"})):
            env = run56.step_env(run56.DEFAULT_HOOK, t / f"{how}.jsonl", cls.cache, extra)
            argv = run56.triage_argv(xr, cls.subject, cls.msg, how=how, channel="email", model=None)
            p = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=str(xr))
            cls.verdicts[how] = json.loads(p.stdout)
            cls.recs[how] = run56.run_one(argv, xr=xr, env=env, parry=cls.parry)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_dead_host_is_named_and_shown_and_the_skip_is_not(self):
        dead, nollm = self.recs["dead"], self.recs["nollm"]
        self.assertEqual((dead["leaf_failure_named"], dead["card_leaf_failure_shown"]), (True, True))
        self.assertEqual((nollm["leaf_failure_named"], nollm["card_leaf_failure_shown"]), (False, False))
        for r in (dead, nollm):
            self.assertIsNone(r["error"])
            self.assertIn(r["band"], ("benign", "suspicious", "likely-scam"))
            self.assertEqual((r["leaf_ok"], r["leaf_output"], r["model"]), (False, None, None), "the leaf did not run")
            self.assertRegex(r["run_id"], r"^\d{8}-\d{6}-[a-z0-9-]+$")

    def test_the_card_is_read_on_its_own(self):
        v = json.loads(json.dumps(self.verdicts["dead"]))
        v["verdict"]["unresolved"] = [u for u in v["verdict"]["unresolved"] if not u.startswith("persona leaf: ")]
        m = run56.measures(v, self.parry)
        self.assertEqual((m["leaf_failure_named"], m["card_leaf_failure_shown"]), (False, False))
        card, persona, score = self.parry.mods()
        for fake_card in ("Checks not run\n   · domain age for x: offline\n",          # named, not on the card
                          "Notes (not scored)\n   · persona leaf: model lock: URLError: refused\n"):
            p = run56.Parry(self.parry.dir)
            p._mods = (types.SimpleNamespace(render=lambda v, c=fake_card: c), persona, score)
            m = run56.measures(self.verdicts["dead"], p)
            self.assertEqual((m["leaf_failure_named"], m["card_leaf_failure_shown"]), (True, False))

    def test_indicators_keep_the_id_only(self):
        v = json.loads(json.dumps(self.verdicts["nollm"]))
        v["verdict"]["indicators"] = [{"id": "rule_b", "tier": "x", "points": 5, "evidence": [MARK], "data": {}},
                                      {"id": "rule_a", "tier": "x", "points": 5, "evidence": [MARK], "data": {}}]
        m = run56.measures(v, self.parry)
        self.assertEqual(m["indicators"], ["rule_a", "rule_b"])
        self.assertNotIn(MARK, json.dumps(m))

    def test_leaf_labels_are_the_closed_enums(self):
        ok = {"ok": True, "output": {"approach": "personal", "asks": ["none", "click_link"], "confidence": "low"}}
        self.assertEqual(self.parry.labels(ok), ok["output"])
        with self.assertRaises(ValueError):
            self.parry.labels({"ok": True, "output": {"approach": MARK, "asks": ["none"], "confidence": "low"}})
        with self.assertRaises(ValueError):
            self.parry.labels({"ok": True, "output": {"approach": "personal", "asks": [MARK], "confidence": "low"}})

    def test_no_record_holds_message_text(self):
        for r in self.recs.values():
            self.assertNotIn(MARK, json.dumps(r))
            self.assertNotIn("example.org", json.dumps(r))

    def args(self, hook):
        return run56.parse_args(["--plan", "registered", "--build", "b", "--base-dir", "d", "--out", "o",
                                 "--cache", str(self.cache), "--skip-canary", "--hook-dir", str(hook)])

    def preflight(self, hook):
        bases, build = fixture(Path(tempfile.mkdtemp(dir=self.t)))
        return run56.preflight(self.args(hook), self.xr, build, bases, Path(tempfile.mkdtemp(dir=self.t)) / "o",
                               hook, Path("/nonexistent"))

    def test_s6_the_hooks_window_ends_before_the_first_message_field(self):
        pf = self.preflight(run56.DEFAULT_HOOK)
        lr = pf["leaf_request"]
        self.assertEqual(lr["hook_windows"], {"request": 800, "send": 200})
        self.assertGreaterEqual(lr["prefix"]["request"], 800 + run56.MARGIN)
        self.assertGreaterEqual(lr["prefix"]["send"], 200 + run56.MARGIN)
        wide = Path(tempfile.mkdtemp(dir=self.t)) / "hook"
        wide.mkdir()
        src = (run56.DEFAULT_HOOK / "sitecustomize.py").read_text()
        (wide / "sitecustomize.py").write_text(src.replace("repr(a)[:800]", f"repr(a)[:{lr['prefix']['request']}]"))
        with self.assertRaisesRegex(run56.Refuse, "margin"):
            self.preflight(wide)
        (wide / "sitecustomize.py").write_text(src.replace("args[1][:200]", "args[1]"))
        with self.assertRaisesRegex(run56.Refuse, "cannot read"):
            self.preflight(wide)


# ── the draw ─────────────────────────────────────────────────────────────────────────────────────────────────────
class Selection(unittest.TestCase):
    def corpus(self, root: Path):
        """A synthetic corpus: two ids more than each category needs, and one excluded id in seven of the pools."""
        man, P, R = {}, [], []
        n = 300
        spec = [("lure", "email", 3, "suspicious", "benign"), ("lure", "email", 3, "suspicious", "suspicious"),
                ("lure", "email", 3, "likely-scam", "suspicious"), ("lure", "sms", 3, "suspicious", "benign"),
                ("benign", "email", 0, "suspicious", "benign"), ("benign", "email", 3, "benign", "benign"),
                ("benign", "email", 0, "benign", "benign"), ("benign", "whatsapp", 0, "benign", "benign")]
        ks = [6, 3, 1, 2, 4, 4, 2, 2]
        (root / "messages").mkdir(parents=True)
        excl = iter(sorted(select56.EXCLUDED))
        for (kind, ch, pts, pb, rb), k in zip(spec, ks):
            ids = [f"X{n + j}" for j in range(k + 2)] + [x for x in [next(excl, None)] if x]
            n += k + 2
            for i in ids:
                f = f"{i}.{'eml' if ch == 'email' else 'txt'}"
                man[i] = {"file": f, "kind": kind, "channel": ch}
                (root / "messages" / f).write_bytes(f"harmless {i}\r\n".encode())
                P.append({"id": i, "band": pb, "persona_points": pts})
                R.append({"id": i, "band": rb, "persona_points": 0})
        (root / "MANIFEST.json").write_text(json.dumps(man))
        for name, rows in (("P.jsonl", P), ("R.jsonl", R)):
            (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return root

    def test_the_draw_is_seeded_complete_and_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as t:
            c = self.corpus(Path(t) / "corpus")
            outs = []
            for k in (1, 2):
                o = Path(t) / f"sel{k}"
                rc, _ = quiet(select56.main, ["--out", str(o), "--corpus", str(c), "--p-results", str(c / "P.jsonl"),
                                              "--r-results", str(c / "R.jsonl")])
                self.assertEqual(rc, 0)
                outs.append(o)
            listing = json.loads((outs[0] / "bases.json").read_text())
            self.assertEqual((outs[0] / "bases.json").read_text(), (outs[1] / "bases.json").read_text())
            self.assertEqual([b["category"] for b in listing],
                             ["L1"] * 6 + ["L2"] * 3 + ["L3"] + ["L4"] * 2 + ["G1"] * 4 + ["G2"] * 4 + ["G3"] * 2
                             + ["G4"] * 2)
            self.assertFalse({b["id"] for b in listing} & select56.EXCLUDED)
            self.assertEqual(len({b["id"] for b in listing}), 24)
            self.assertEqual({b["kind"] for b in listing[:12]}, {"lure"})
            self.assertEqual({b["kind"] for b in listing[12:]}, {"genuine"})
            for b in listing:
                src = (c / "messages" / b["file"]).read_bytes()
                self.assertEqual((outs[0] / b["file"]).read_bytes(), src)
                self.assertEqual(b["sha256"], sha(src))
            self.assertEqual(len(list(outs[0].iterdir())), 25)
            rc, _ = quiet(select56.main, ["--out", str(outs[0]), "--corpus", str(c), "--p-results",
                                          str(c / "P.jsonl"), "--r-results", str(c / "R.jsonl")])
            self.assertEqual(rc, 2, "a folder that is not empty is refused")

    @unittest.skipUnless((select56.DEFAULT_CORPUS / "MANIFEST.json").is_file(), "needs exp_055's corpus")
    def test_the_real_pools_are_the_ones_the_brief_counted(self):
        man = json.loads((select56.DEFAULT_CORPUS / "MANIFEST.json").read_text())
        picks, pools = select56.draw(man, select56.results(select56.DEFAULT_P), select56.results(select56.DEFAULT_R))
        self.assertEqual(list(pools.values()), [15, 44, 5, 7, 13, 23, 50, 17])
        self.assertEqual(" ".join(i for i, _ in picks),                       # the seeded draw, as run 10 made it
                         "X191 X122 X003 X098 X153 X208 X095 X158 X117 X111 X060 X097 "
                         "X185 X034 X064 X047 X192 X101 X006 X010 X167 X160 X049 X144")


if __name__ == "__main__":
    unittest.main(verbosity=2)
