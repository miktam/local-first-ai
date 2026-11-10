#!/usr/bin/env python3
"""exp_056 runner: the registered sequence (Amendment 2, "Runs"), every step under exp_054's audit hook.

Non-attack scaffolding (Claude's side of the split): this file never contains or reads payload or message text. It
hands file paths to `exposure.py triage` as a subprocess, parses the JSON verdict in memory and keeps measures only.
The payloads and the builder (build56.py) are Andrei's; this file hashes their bytes for meta.json and nothing else.

  python3.14 kit/run56.py --plan registered --build BUILD --base-dir BASES --out OUT --cache CACHE [--skip-canary]

Order (one record per run in OUT/results.jsonl, sorted by id within a step, resumable by key = <step>:<id>):
  canary    exp_054's kit/control_c2.py under the hook, into OUT/audit/canary.jsonl (--skip-canary: not run); the
            run stops (exit 2) unless the log shows a resolution, an HTTP request and a spawned dig
  bases1    every base, leaf on                 the reference for H1, H2 and C2
  variants  every variant, leaf on              H1, H2
  bases2    every base again, leaf on           C2
  nollm     every base, --no-llm                H5's reference
  dead      every base, model host dead         H5 (class F: TRIAGE_OLLAMA at --dead-host, a closed loopback port)
Each run is `exposure.py triage <subject copy> <file> --offline --json`, plus `--channel <channel>` for a base whose
channel is not email and for its variants, exactly as exp_055's P arm (kit/arms55.py); one second apart; with
PYTHONPATH=<hook dir> and AUDIT54_LOG=OUT/audit/<step>.jsonl. The subject copy is built once, under the hook too
(OUT/audit/setup.jsonl), and the seeded registry cache is copied once to OUT/cache (TRIAGE_CACHE).

Pre-flight, before anything is written; any failure is exit 2 and one line on stderr that never echoes a value it
has not validated. BASES/bases.json ({file, id, kind: lure|genuine, channel, category, sha256}; select56.py writes
it): every listed base's bytes match its sha256. BUILD/manifest.json (build56.py's, untrusted): every name and payload
label is plain, every base it names is listed, no variant or (base, payload) pair twice, every listed base has the
same set of payloads. A resume must find the same manifest, bases and kit hashes in OUT/meta.json. The leaf's
request, built by Parry's own code, must keep its message fields clear of the audit hook's windows by 100.

A record holds measures, never text: no stdout, no stderr, no `unresolved` line, no model or message free text.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import score56   # the kit's own log reader, for the canary check

WS = Path.home() / ".openclaw/workspace"
E054 = Path.home() / "REPOS/local-first-ai/tasks/chronos/exp_054_exposure_review_egress"
DEFAULT_XR = WS / "tools/exposure-report"
DEFAULT_SUBJECT = WS / "tasks/exposure-report/demo"
DEFAULT_HOOK = E054 / "kit/hook"
DEFAULT_CANARY = E054 / "kit/control_c2.py"
WS_TOOLS = {"parry": WS / "tools/parry", "exposure_report": WS / "tools/exposure-report"}
KIT = Path(__file__).resolve().parent
KIT_FILES = ("run56.py", "score56.py", "select56.py", "build56.py")
PY = "python3.14"
GAP_S = 1.0                                   # triage run folders are named to the second
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
LEAF_LABELS = ("approach", "asks", "confidence")
RUN_ID_RX = re.compile(r"\d{8}-\d{6}-[a-z0-9-]{1,40}")
NAME_RX = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}")     # a base or variant file name
LABEL_RX = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,39}")    # a payload label
CHANNELS = ("email", "x", "linkedin", "telegram", "whatsapp", "signal", "discord", "sms", "other")  # triage --channel
HOOK_WINDOWS = (("request", re.compile(r"repr\(a\)\[:(\d+)\]")),      # urllib.Request: repr of each argument
                ("send", re.compile(r"args\[1\]\[:(\d+)\]")))         # http.client.send: the raw bytes sent
MARGIN = 100
# The registered sequence: (step, what it runs, how).
PLAN = (("bases1", "bases", "leaf"), ("variants", "variants", "leaf"), ("bases2", "bases", "leaf"),
        ("nollm", "bases", "nollm"), ("dead", "bases", "dead"))


class Refuse(Exception):
    """A pre-flight failure: one line on stderr, exit 2."""


def sha256_bytes(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def plain(rx, v) -> bool:
    return isinstance(v, str) and rx.fullmatch(v) is not None


def normalize_manifest(raw) -> list[dict]:
    """build56's manifest, in whatever shape, -> [{variant, payload, base}]. Accepts a {'variants': [...]} list,
    a {variant: {payload, base}} map, or a {variant: [payload, base]} map. Raises on anything else."""
    rows = raw["variants"] if isinstance(raw, dict) and "variants" in raw else raw
    out = []
    if isinstance(rows, list):
        for r in rows:
            out.append({"variant": r["variant"], "payload": r.get("payload"), "base": r.get("base")})
    elif isinstance(rows, dict):
        for variant, v in rows.items():
            if isinstance(v, dict):
                out.append({"variant": variant, "payload": v.get("payload"), "base": v.get("base")})
            elif isinstance(v, (list, tuple)):
                out.append({"variant": variant, "payload": v[0], "base": v[1] if len(v) > 1 else None})
            else:
                out.append({"variant": variant, "payload": v, "base": None})
    else:
        raise ValueError("expected a list or a map of variants")
    if not out:
        raise ValueError("no variants")
    return out


# ── pre-flight: everything checked before anything is written ─────────────────────────────────────────────────────
def load_bases(base_dir: Path) -> list[dict]:
    """bases.json, each entry checked and each listed base's bytes checked against its sha256. -> sorted by file."""
    path = base_dir / "bases.json"
    try:
        listing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refuse(f"{path}: not a readable list of bases ({type(e).__name__})")
    if not isinstance(listing, list) or not listing:
        raise Refuse(f"{path}: expected a non-empty list of bases")
    names = set()
    for n, b in enumerate(listing, 1):
        where = f"bases.json entry {n}"
        if not isinstance(b, dict) or not plain(NAME_RX, b.get("file")) or not plain(NAME_RX, b.get("id")):
            raise Refuse(f"{where}: the file or the id is not a plain name")
        if b["file"] in names or b["id"] in names - {b["file"]}:
            raise Refuse(f"{where}: {b['file']} is listed twice")
        names |= {b["file"], b["id"]}
        if b.get("kind") not in ("lure", "genuine"):
            raise Refuse(f"{where}: kind must be lure or genuine")
        if b.get("channel") not in CHANNELS:
            raise Refuse(f"{where}: channel must be one of {', '.join(CHANNELS)}")
        f = base_dir / b["file"]
        if not f.is_file():
            raise Refuse(f"{b['file']}: listed in bases.json but missing from {base_dir}")
        if sha256_bytes(f) != b.get("sha256"):
            raise Refuse(f"{b['file']}: bytes do not match the sha256 listed in bases.json")
    return sorted(listing, key=lambda b: b["file"])


def load_variants(build: Path, bases: list[dict]) -> tuple[list[dict], list[str]]:
    """The manifest's variants, each tied to a listed base (named by file or by id); the manifest is untrusted, so
    a refusal names a row by its position and echoes only values already found plain. -> (sorted by variant, the
    payload labels every base has)."""
    path = build / "manifest.json"
    try:
        rows = normalize_manifest(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as e:
        raise Refuse(f"{path}: not a readable variant manifest ({type(e).__name__})")
    by_name = {b["file"]: b for b in bases} | {b["id"]: b for b in bases}
    seen, pairs, out = set(), set(), []
    for n, row in enumerate(rows, 1):
        where = f"manifest.json row {n}"
        if not plain(NAME_RX, row["variant"]):
            raise Refuse(f"{where}: the variant name is not a plain name")
        if not plain(LABEL_RX, row["payload"]):
            raise Refuse(f"{where}: the payload label is not a plain label")
        if not plain(NAME_RX, row["base"]):
            raise Refuse(f"{where}: the base name is not a plain name")
        base = by_name.get(row["base"])
        if base is None:
            raise Refuse(f"{where}: its base {row['base']} is not listed in bases.json")
        if row["variant"] in seen:
            raise Refuse(f"{where}: variant {row['variant']} is listed twice")
        if (base["file"], row["payload"]) in pairs:
            raise Refuse(f"{where}: base {base['file']} has payload {row['payload']} twice")
        if not (build / row["variant"]).is_file():
            raise Refuse(f"{row['variant']}: named by {where} but missing from {build}")
        seen.add(row["variant"])
        pairs.add((base["file"], row["payload"]))
        out.append({"variant": row["variant"], "payload": row["payload"], "base": base})
    sets = {b["file"]: {p for f, p in pairs if f == b["file"]} for b in bases}
    ref_base = bases[0]["file"]
    for f, s in sets.items():
        if s != sets[ref_base]:
            raise Refuse(f"{f}: {len(s)} payload(s), not the same set as {ref_base}'s {len(sets[ref_base])}; "
                         f"every listed base needs the same payloads")
    return sorted(out, key=lambda r: r["variant"]), sorted(sets[ref_base])


def check_dead_host(url: str) -> None:
    """The dead host must be a loopback address with nothing listening (Amendment 2: checked unbound). Nothing is
    sent anywhere else: a host that is not loopback is refused before any connection is tried."""
    u = urllib.parse.urlsplit(url)
    if u.hostname not in LOOPBACK or u.port is None:
        raise Refuse(f"--dead-host {url}: must be a loopback address with a port")
    try:
        with socket.create_connection((u.hostname, u.port), timeout=1):
            pass
    except OSError:
        return
    raise Refuse(f"--dead-host {url}: something is listening there; the dead-model step needs a closed port")


def hook_windows(hook: Path) -> dict:
    """How much of each call's arguments exp_054's hook keeps, read from its own source."""
    src = (hook / "sitecustomize.py").read_text(encoding="utf-8")
    out = {}
    for name, rx in HOOK_WINDOWS:
        found = rx.findall(src)
        if len(found) != 1:
            raise Refuse(f"{hook / 'sitecustomize.py'}: cannot read the hook's {name} window from its source")
        out[name] = int(found[0])
    return out


def parry_dir_of(xr: Path) -> Path | None:
    """The Parry checkout exposure.py itself uses (xrlib.TRIAGE_DIR), or None when it can't say."""
    try:
        p = subprocess.run([PY, "-c", "import xrlib; print(xrlib.TRIAGE_DIR)"], capture_output=True, text=True,
                           cwd=str(xr), timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    d = Path(p.stdout.strip()) if p.returncode == 0 and p.stdout.strip() else None
    return d if d and (d / "triage" / "card.py").is_file() else None


def tool_commit(folder: Path | None, name: str) -> str | None:
    """The last commit touching the workspace's tools/<name> folder, when `folder` is that checkout; else None."""
    if folder is None or folder.resolve() != WS_TOOLS[name].resolve():
        return None
    p = subprocess.run(["git", "-C", str(WS), "log", "-1", "--format=%H", "--", f"tools/{WS_TOOLS[name].name}"],
                       capture_output=True, text=True)
    h = p.stdout.strip()
    return h if re.fullmatch(r"[0-9a-f]{40}", h) else None


# ── one run, measured ─────────────────────────────────────────────────────────────────────────────────────────────
class Parry:
    """Parry's own card, leaf and unresolved-line mapping, imported read-only from the checkout exposure.py uses."""

    def __init__(self, parry_dir: Path | None):
        self.dir, self._mods = parry_dir, None

    def mods(self):
        if self._mods is None:
            if self.dir is None:
                raise ModuleNotFoundError("Parry's checkout")
            if str(self.dir) not in sys.path:
                sys.path.append(str(self.dir))
            from triage import card, persona, score
            self._mods = card, persona, score
        return self._mods

    def leaf_failure(self, line) -> bool:
        """An unresolved line (or a card entry) that names a failure of the leaf, the --no-llm skip excepted.
        Parry's own mapping: triage.score.unresolved_kind -> leaf_* kinds; 'leaf_skipped' is the --no-llm choice."""
        kind = self.mods()[2].unresolved_kind(line)
        return kind.startswith("leaf_") and kind != "leaf_skipped"

    def card_shows_leaf_failure(self, verdict) -> bool:
        """Parry's terminal card, rendered in memory: does 'Checks not run' list a failure of the leaf? Only the
        boolean leaves this function."""
        lines = self.mods()[0].render(verdict).splitlines()
        if "Checks not run" not in lines:
            return False
        entries = []
        for line in lines[lines.index("Checks not run") + 1:]:
            if line.startswith("   · "):
                entries.append(line[len("   · "):])
            elif line.startswith("     ") and entries:      # a wrapped entry continues
                entries[-1] += " " + line.strip()
            else:
                break
        return any(self.leaf_failure(e) for e in entries)

    def labels(self, leaf) -> dict | None:
        """The leaf's three closed labels, each checked against Parry's own enums; None when the leaf did not
        answer. A label outside the schema raises: it would be free text in the record."""
        if not leaf.get("ok") or not isinstance(leaf.get("output"), dict):
            return None
        persona, out = self.mods()[1], leaf["output"]
        lab = {k: out.get(k) for k in LEAF_LABELS}
        if (lab["approach"] not in persona.APPROACHES or lab["confidence"] not in persona.CONFIDENCE
                or not isinstance(lab["asks"], list) or not all(a in persona.ASKS for a in lab["asks"])):
            raise ValueError("a leaf label outside Parry's schema")
        return lab


_STOP = {"armed": False, "data": None, "hooked": False}


class _Stopped(Exception):
    pass


def _stop_request(event, args):
    if _STOP["armed"] and event == "urllib.Request":
        _STOP["armed"], _STOP["data"] = False, args[1]
        raise _Stopped("stopped before any connection")


def leaf_request_prefix(parry: Parry, model: str | None) -> dict:
    """How much of the leaf's chat request comes before its first field from the message, measured with Parry's
    own code: persona.run on a message whose fields are sentinels, the model lock stubbed (it only adds facts to
    the result, not to the request), and the request stopped by an audit hook before urllib opens any connection
    (its host is a closed loopback port besides). Counted to the first field substituted into the prompt, the
    channel included. -> {"request": characters of the bytes' repr (what the hook keeps of urllib.Request),
    "send": raw bytes (what it keeps of http.client.send)}."""
    persona = parry.mods()[1]
    msg = {"channel": "S6CHANNEL", "sender_name": "S6SENDERNAME", "sender_addr": "S6SENDERADDR",
           "claim": "S6CLAIM", "text": "S6TEXT"}
    if not _STOP["hooked"]:
        sys.addaudithook(_stop_request)
        _STOP["hooked"] = True
    lock, _STOP["data"] = persona.lock, None
    persona.lock = lambda *a, **k: {}
    _STOP["armed"] = True
    try:
        persona.run(msg, model=model or persona.DEFAULT_MODEL, host="http://127.0.0.1:9", timeout=5)
    finally:
        persona.lock, _STOP["armed"] = lock, False
    data = _STOP["data"]
    if not isinstance(data, bytes):
        raise Refuse("cannot capture the leaf's request from Parry's code; the hook's margin is unchecked")
    r = repr(data)
    at_r = [r.find(s) for s in msg.values() if r.find(s) >= 0]
    at_b = [data.find(s.encode()) for s in msg.values() if data.find(s.encode()) >= 0]
    if not at_r or not at_b:
        raise Refuse("cannot find the message fields in the leaf's request; the hook's margin is unchecked")
    return {"request": min(at_r), "send": min(at_b)}


def measures(d: dict, parry: Parry) -> dict:
    """exposure.py's JSON result -> the record's measures. Nothing here is text from the message or the model."""
    v = d["verdict"]
    leaf = v.get("persona") or {}
    model = None
    if leaf.get("digest"):                         # the model was locked and called; null when the leaf did not run
        model = {k: leaf.get(k) for k in ("digest", "ollama_version", "quantization", "prompt_sha256", "options")}
    run_id = d.get("run_id")
    return {"band": v.get("band"), "score": v.get("score"), "persona_points": v.get("persona_points"),
            "complete": v.get("complete"), "leaf_ok": leaf.get("ok") is True, "leaf_output": parry.labels(leaf),
            "indicators": sorted(str(i["id"]) for i in v.get("indicators") or []),
            "leaf_failure_named": any(parry.leaf_failure(u) for u in v.get("unresolved") or []),
            "card_leaf_failure_shown": parry.card_shows_leaf_failure(v), "model": model,
            "run_id": run_id if isinstance(run_id, str) and RUN_ID_RX.fullmatch(run_id) else None}


EMPTY = {k: None for k in ("band", "score", "persona_points", "complete", "leaf_ok", "leaf_output", "indicators",
                           "leaf_failure_named", "card_leaf_failure_shown", "model", "run_id")}


def triage_argv(xr: Path, subject: Path, file: Path, *, how: str, channel: str, model: str | None) -> list[str]:
    """One run's command line: exp_055's P arm (kit/arms55.py), with --no-llm for the rules-only step."""
    argv = [PY, str(xr / "exposure.py"), "triage", str(subject), str(file), "--offline", "--json"]
    if how == "nollm":
        argv.append("--no-llm")
    if channel != "email":
        argv += ["--channel", channel]
    if model:
        argv += ["--model", model]
    return argv


def run_one(argv: list[str], *, xr: Path, env: dict, parry: Parry) -> dict:
    t0, rc = time.time(), None
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env, cwd=str(xr))
        rc = p.returncode
        rec = {"rc": rc, "seconds": round(time.time() - t0, 2), **measures(json.loads(p.stdout), parry), "error": None}
    except Exception as e:                         # the class name only: no stderr, no stdout, no message
        rec = {"rc": rc, "seconds": round(time.time() - t0, 2), **EMPTY, "error": type(e).__name__}
    return rec


# ── the run ───────────────────────────────────────────────────────────────────────────────────────────────────────
def step_env(hook: Path, log: Path, cache: Path, extra: dict | None = None) -> dict:
    return {**os.environ, "PYTHONPATH": str(hook), "AUDIT54_LOG": str(log), "TRIAGE_CACHE": str(cache),
            **(extra or {})}


def canary_status(log: Path) -> str:
    """'skipped' (no log), 'ran' (the log shows a resolution, an HTTP request and a spawned dig) or 'failed'."""
    if not log.is_file():
        return "skipped"
    return "ran" if all(score56.canary_acts(log).values()) else "failed"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True, choices=["registered"], help="the registered sequence (Amendment 2)")
    ap.add_argument("--build", required=True, help="build56.py's output dir: the variant files and manifest.json")
    ap.add_argument("--base-dir", required=True, help="the base messages and bases.json (select56.py's output)")
    ap.add_argument("--out", required=True, help="run folder: results.jsonl, meta.json, audit/ (outside any repo)")
    ap.add_argument("--cache", required=True, help="seeded registry cache, copied once to OUT/cache (TRIAGE_CACHE)")
    ap.add_argument("--skip-canary", action="store_true", help="do not run the canary (tests and checks only)")
    ap.add_argument("--canary", default=str(DEFAULT_CANARY), help="the canary script (exp_054's control_c2.py)")
    ap.add_argument("--hook-dir", default=str(DEFAULT_HOOK), help="exp_054's audit hook (sitecustomize.py)")
    ap.add_argument("--payloads", help="the frozen payload folder: its files' sha256 go into meta.json")
    ap.add_argument("--xr-dir", default=str(DEFAULT_XR), help="exposure-report checkout")
    ap.add_argument("--subject", default=str(DEFAULT_SUBJECT), help="demo subject folder to copy")
    ap.add_argument("--model", help="model tag for the leaf (default: Parry's gemma4:26b)")
    ap.add_argument("--dead-host", default="http://127.0.0.1:9", help="the dead step's model host (closed port)")
    return ap.parse_args(argv)


def preflight(a, xr, build, base_dir, out, hook, canary) -> dict:
    """Everything checked before anything is written. -> what the run needs. Raises Refuse."""
    bases = load_bases(base_dir)
    variants, payloads = load_variants(build, bases)
    if not (hook / "sitecustomize.py").is_file():
        raise Refuse(f"{hook / 'sitecustomize.py'}: the audit hook is missing")
    windows = hook_windows(hook)
    if not a.skip_canary and not canary.is_file():
        raise Refuse(f"{canary}: the canary is missing")
    if not (xr / "exposure.py").is_file():
        raise Refuse(f"{xr / 'exposure.py'}: no exposure-report checkout there")
    for folder, what in ((a.cache, "the seeded cache"), (a.subject, "the demo subject"),
                         (a.payloads, "the payload folder")):
        if folder is not None and not Path(folder).is_dir():
            raise Refuse(f"{folder}: {what} is missing")
    check_dead_host(a.dead_host)
    parry = Parry(parry_dir_of(xr))
    prefix = None
    if parry.dir is not None:                      # without Parry no run can be measured; every record says so
        prefix = leaf_request_prefix(parry, a.model)
        for name, w in windows.items():
            if prefix[name] < w + MARGIN:
                raise Refuse(f"the leaf's request keeps {prefix[name]} fixed characters before its first message "
                             f"field; the hook's {name} window is {w}, and the margin must be {MARGIN}")
    pinned = {"manifest_sha256": sha256_bytes(build / "manifest.json"),
              "bases": {b["file"]: b["sha256"] for b in bases},
              "kit": {f: sha256_bytes(KIT / f) if (KIT / f).is_file() else None for f in KIT_FILES}}
    meta_path = out / "meta.json"
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text())
        except (OSError, ValueError) as e:
            raise Refuse(f"{meta_path}: unreadable ({type(e).__name__}); start a new --out")
        differ = [k for k in pinned if old.get(k) != pinned[k]]
        if differ:
            raise Refuse(f"{meta_path}: the {', '.join(differ)} differ from the run being resumed; start a new --out")
    expected = {step: [f"{step}:{v['variant']}" for v in variants] if what == "variants" else
                [f"{step}:{b['file']}" for b in bases] for step, what, _ in PLAN}
    return {"bases": bases, "variants": variants, "payloads": payloads, "parry": parry, "pinned": pinned,
            "expected": expected,
            "leaf_request": None if prefix is None else {"prefix": prefix, "hook_windows": windows, "margin": MARGIN}}


def main(argv=None) -> int:
    a = parse_args(argv)
    # Absolute paths throughout: triage runs with cwd=xr, so a relative path would resolve under the checkout.
    xr, build, base_dir = Path(a.xr_dir).resolve(), Path(a.build).resolve(), Path(a.base_dir).resolve()
    out, hook, canary = Path(a.out).resolve(), Path(a.hook_dir).resolve(), Path(a.canary).resolve()
    try:
        pf = preflight(a, xr, build, base_dir, out, hook, canary)
    except Refuse as e:
        print(f"run56: {e}; nothing ran", file=sys.stderr)
        return 2
    except Exception as e:                         # any other pre-flight failure: still one line, still exit 2
        print(f"run56: the pre-flight failed ({type(e).__name__}); nothing ran", file=sys.stderr)
        return 2
    bases, variants, parry = pf["bases"], pf["variants"], pf["parry"]

    (out / "audit").mkdir(parents=True, exist_ok=True)
    meta_path, res_path, canary_log = out / "meta.json", out / "results.jsonl", out / "audit" / "canary.jsonl"
    old = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    meta = {"plan": a.plan, "canary": None, "started": old.get("started") or now(), "finished": None,
            "argv": sys.argv if argv is None else ["run56.py", *argv], **pf["pinned"],
            "tools": {"parry": tool_commit(parry.dir, "parry"), "exposure_report": tool_commit(xr, "exposure_report")},
            "payloads": ({f.name: sha256_bytes(f) for f in sorted(Path(a.payloads).iterdir()) if f.is_file()}
                         if a.payloads else None),
            "hook_sha256": sha256_bytes(hook / "sitecustomize.py"),
            "leaf_request": pf["leaf_request"], "payload_labels": pf["payloads"], "expected": pf["expected"]}

    def write_meta():
        meta["canary"] = canary_status(canary_log)
        meta_path.write_text(json.dumps(meta, indent=1) + "\n")

    write_meta()
    # A fresh copy of the subject, built once under the hook; one copy of the seeded cache.
    subject, cache = out / "subject", out / "cache"
    if not cache.exists():
        shutil.copytree(a.cache, cache)
    if not subject.exists():
        shutil.copytree(a.subject, subject, ignore=shutil.ignore_patterns("out", ".lock"))
        p = subprocess.run([PY, str(xr / "exposure.py"), "build", str(subject)], capture_output=True, cwd=str(xr),
                           env=step_env(hook, out / "audit" / "setup.jsonl", cache))
        if p.returncode != 0:
            shutil.rmtree(subject)
            print(f"run56: building the subject copy failed (rc {p.returncode})", file=sys.stderr)
            return 1
    # The canary, before the first step, under the same hook (never again once its log exists).
    if not a.skip_canary and not canary_log.exists():
        subprocess.run([PY, str(canary)], capture_output=True, timeout=120, cwd=str(out),
                       env=step_env(hook, canary_log, cache))
    write_meta()
    if not a.skip_canary and meta["canary"] != "ran":
        missing = [k for k, ok in score56.canary_acts(canary_log).items() if not ok]
        print(f"run56: the canary's log lacks {', '.join(missing)}: the hook is not shown to be live; no step ran",
              file=sys.stderr)
        return 2

    done = {json.loads(l)["key"] for l in res_path.read_text().splitlines() if l.strip()} if res_path.exists() else set()
    first = True
    with open(res_path, "a") as f:
        for step, what, how in PLAN:
            items = ([{"id": v["variant"], "role": "variant", "file": build / v["variant"], "payload": v["payload"],
                       "base": v["base"]} for v in variants] if what == "variants" else
                     [{"id": b["file"], "role": "base", "file": base_dir / b["file"], "payload": None, "base": b}
                      for b in bases])
            env = step_env(hook, out / "audit" / f"{step}.jsonl", cache,
                           {"TRIAGE_OLLAMA": a.dead_host} if how == "dead" else None)
            for it in sorted(items, key=lambda i: i["id"]):
                key = f"{step}:{it['id']}"
                if key in done:
                    continue
                if not first:
                    time.sleep(GAP_S)
                first = False
                argv_ = triage_argv(xr, subject, it["file"], how=how, channel=it["base"]["channel"], model=a.model)
                r = run_one(argv_, xr=xr, env=env, parry=parry)
                rec = {"key": key, "step": step, "id": it["id"], "role": it["role"], "kind": it["base"]["kind"],
                       "payload": it["payload"], "base": it["base"]["file"], "sha256": sha256_bytes(it["file"]), **r}
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                f.flush()
                print(step, it["id"], rec["band"], rec["error"] or "", rec["seconds"], flush=True)
    meta["finished"] = now()
    write_meta()
    print("done", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
