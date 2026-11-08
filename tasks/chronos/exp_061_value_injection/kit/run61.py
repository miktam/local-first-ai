#!/usr/bin/env python3
"""
run61.py runs exp_061's two arms (HYPOTHESIS.md, "Arms") over its three parts on the mini's
Ollama daemon, one model at a time.

    Q  hf.co/bartowski/Qwen3.8-27B-GGUF:Q4_K_M                 the base
    A  hf.co/bartowski/NousResearch_Arete-27B-GGUF:Q4_K_M      the fine-tune

Parts: h1 = the translation pilot's sample.json (40 units, sha256 checked against the
registration); h2 = the h2 units of kit/sample61.json; h3 = its h3 prompts.

Contracts. H1 and H2 translate through the pilot's own code, imported, not copied: the
pilot's run.py (petukhov_system, contexts) and translate.py (Ctx, warmup, translate_unit, so
ollama_translate, the validator and the one retry with the reason). Mamardashvili units get
translate.py's system prompt with glossary rev. 2; Petukhov units get pipeline.py's MT_SYSTEM
with its JSON rule replaced by the plain-text rule, as in the pilot. H3 sends each prompt of
the vendor's eval verbatim as the only (user) message, with no system prompt.

Settings (the registration's): every /api/chat request states num_ctx 8192, temperature 0
and num_predict 2048, and think false (unless the model rejects `think`, as translate.py
handles it). translate.py sets the first two itself and no num_predict, so a urllib handler
installed here adds num_predict 2048 to each /api/chat request, checks the other two, and
records each request's options with the response's counters (token counts, done_reason,
durations); translate.py is unchanged. Each arm is unloaded with keep_alive 0 after its
parts; the bots' gemma4:26b is reloaded with keep_alive -1 at the end, also on an error or
Ctrl-C.

The gate. Before any request, a real run exits unless the pilot's contracts are the
registration's frozen inputs: prompt hashes 8290bb7c2223 (Mamardashvili) and c99ba298d5ff
(Petukhov), glossary a3af943b0655, and the pilot README's sha256 5f60ac8a…. A dry run records
any mismatch under contracts.frozen_mismatch.

Failures. An H3 prompt gets three attempts, as translate_unit gives a unit: a connection
error (a socket dropped mid-request included), an HTTP 5xx, an empty or broken answer is
retried, a 4xx is not. When the daemon cannot be reached after the attempts (a translation's
or an H3 prompt's), the unit gets no row and the run stops with the --resume line; so does a
socket dropped inside the pilot's translate_unit, which retries URLError only.
`--resume <folder>` continues a stopped run: the folder's parts, arms, limit and units (their
sha256 checked), the same contracts, Ollama version and blobs, results.jsonl appended, every
(part, arm, id) it holds skipped, an arm with nothing left not loaded; the session goes under
meta.json's `resumes`.

Output, runs/<UTC stamp>[-smoke|-dry]/ (or --out):
- meta.json: the serving-stack row (Ollama version; per arm the tag digest, /api/show's
  blob sha256 from the FROM line and from the blob's bytes, details, template sha256, and
  /api/ps's runner and context after warm-up; flash attention and KV cache type are not
  exposed by the API, so they are recorded from workspace CLAUDE.md's production env as of
  2026-08-30, with that date), OS build, thermal pressure at start and end (notifyutil,
  readable without sudo; powermetrics needs sudo and is never called), seed, arms, settings,
  the units' sources with sha256, the contracts' prompt hashes, dry_run.
- results.jsonl: one row per unit per arm per part: part, arm, model, corpus or category, id,
  the request's options, think, the answer, attempts, error, prompt_eval_count, eval_count,
  done_reason, wall_s, and every call of a retried unit under `calls`.
- ps_before.json, ps_after.json (not in a dry run).

--dry-run writes meta.json and an empty results.jsonl and sends no model request: at most
GET /api/version and GET /api/tags, only if the daemon answers, listed in meta.json under
`requests_made`. It runs under python3 (3.9) too, where it records that the pilot's
contracts cannot load; every other mode needs python3.14.

    python3.14 -B kit/run61.py --dry-run
    python3.14 -B kit/run61.py --limit 1                 # smoke test, runs/<stamp>-smoke
    python3.14 -B kit/run61.py                           # full run: both arms, h1,h2,h3
    python3.14 -B kit/run61.py --parts h1 --arms A       # a subset
    python3.14 -B kit/run61.py --resume runs/<stamp>     # continue a stopped run
"""
import argparse
import hashlib
import http.client
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
import urllib.response
from pathlib import Path

sys.dont_write_bytecode = True  # never leave __pycache__ in the pilot's folders

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PILOT = Path("/Users/miktam02/repos/library/tasks/mt-model-pilot")
PILOT_SAMPLE_SHA256 = "d46a40a021fd9df5f520ef99115cdcf0a3be7556fad2a7f47e126874fb5a114a"
# The registration's frozen inputs: the pilot README (the contracts and rubric) and the prompt
# hashes the pilot's Q27 row was made under (pilot run 20261105T054758Z, meta.json).
PILOT_README_SHA256 = "5f60ac8a484bdff09bfb187de5633b63e3d180cb7bd74849e616497718235ff9"
FROZEN_CONTRACTS = {"mamardashvili_prompt_sha": "8290bb7c2223", "petukhov_prompt_sha": "c99ba298d5ff",
                    "glossary_sha": "a3af943b0655", "pilot_readme_sha256": PILOT_README_SHA256}
# Not exposed by the Ollama API; the daemon's plist and environment are Andrei's and not read.
SERVING_ENV = {"flash_attention": "OLLAMA_FLASH_ATTENTION=0", "kv_cache_type": "no OLLAMA_KV_CACHE_TYPE",
               "source": "workspace CLAUDE.md, production env as of 2026-08-30", "as_of": "2026-08-30"}
SAMPLE61 = HERE / "sample61.json"

SEED = 20261108
ARMS = {"Q": "hf.co/bartowski/Qwen3.8-27B-GGUF:Q4_K_M",
        "A": "hf.co/bartowski/NousResearch_Arete-27B-GGUF:Q4_K_M"}
BOT_MODEL = "gemma4:26b"
NUM_CTX, TEMPERATURE, THINK, NUM_PREDICT = 8192, 0, False, 2048
OPTIONS = {"temperature": TEMPERATURE, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}
PARTS = ("h1", "h2", "h3")
CORPUS = {"mm": "mamardashvili", "pk": "petukhov"}          # the pilot's codes → this kit's names
PILOT_CODE = {v: k for k, v in CORPUS.items()}
COUNTERS = ("done_reason", "prompt_eval_count", "eval_count", "total_duration", "load_duration",
            "prompt_eval_duration", "eval_duration")


def daemon_host():
    h = os.environ.get("OLLAMA_HOST", "").strip() or "127.0.0.1:11434"
    return (h if "://" in h else "http://" + h).rstrip("/")


HOST = daemon_host()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def warn(msg):
    print(f"[{time.strftime('%H:%M:%S')}] WARN {msg}", file=sys.stderr, flush=True)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def get(path, timeout=10):
    with urllib.request.urlopen(HOST + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path, payload, timeout=900):
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ── the machine ──────────────────────────────────────────────────────────────

def _cmd(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def os_row():
    mem = _cmd(["sysctl", "-n", "hw.memsize"])
    return {"product_version": platform.mac_ver()[0] or None, "build": _cmd(["sysctl", "-n", "kern.osversion"]),
            "hw_model": _cmd(["sysctl", "-n", "hw.model"]), "chip": _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memsize_bytes": int(mem) if mem and mem.isdigit() else None, "python": sys.version.split()[0]}


def thermal():
    """Readable without sudo: notifyutil's thermal pressure level (0 nominal … 4). No sysctl key
    carries thermal pressure on this mini (sysctl -a, macOS 27.2, M4 Pro, read 2026-11-08); the
    Intel key is tried and recorded as null. powermetrics needs sudo and is not called."""
    level = _cmd(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"])
    try:
        level = int(level.split()[-1]) if level else None
    except ValueError:
        level = None
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "notifyutil_thermalpressurelevel": level,
            "sysctl_machdep_xcpm_cpu_thermal_level": _cmd(["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"]),
            "powermetrics": None,
            "note": "powermetrics needs sudo and is not called"}


# ── the pilot's contracts, imported ──────────────────────────────────────────

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pilot_run():
    """The pilot's run.py as a module: petukhov_system(), contexts(), and translate.py as .T.
    translate.py needs Python 3.10+ (it is run with python3.14 on the mini)."""
    if sys.version_info < (3, 10):
        raise RuntimeError(f"the pilot's translate.py needs python3.14; this is {sys.version.split()[0]}")
    return load_module("mt_pilot_run", PILOT / "run.py")


def contracts_row(pr):
    pk_system = pr.petukhov_system()
    probe = pr.contexts("probe", pk_system)
    tr = PILOT.parents[1] / "tasks" / "mamardashvili-ingestion" / "translate.py"
    return {"mamardashvili_prompt_sha": probe["mm"].prompt_sha, "glossary_sha": probe["mm"].glossary_sha,
            "petukhov_prompt_sha": probe["pk"].prompt_sha, "contract": pr.T.CONTRACT,
            "translate_py_sha256": sha256_bytes(tr.read_bytes()),
            "pilot_run_py_sha256": sha256_bytes((PILOT / "run.py").read_bytes()),
            "pilot_readme_sha256": sha256_bytes((PILOT / "README.md").read_bytes()),
            "h3": "the item's prompt verbatim as the only user message; no system prompt"}


def frozen_mismatch(contracts):
    """The registration's frozen contracts that this library no longer has, as 'key: got ≠ frozen'."""
    return [f"{k}: {contracts.get(k)} ≠ {v}" for k, v in FROZEN_CONTRACTS.items() if contracts.get(k) != v]


class RunAborted(Exception):
    """The daemon could not be reached after the retries: the unit gets no row, the run stops,
    and --resume picks it up."""


# ── units ────────────────────────────────────────────────────────────────────

def load_units(parts, sample61, strict):
    """→ ({part: [unit]}, {part: source row}). strict: exit when a part's units are missing."""
    units, sources = {}, {}
    if "h1" in parts:
        raw = (PILOT / "sample.json").read_bytes()
        digest = sha256_bytes(raw)
        if digest != PILOT_SAMPLE_SHA256:
            sys.exit(f"the pilot's sample.json has sha256 {digest}, the registration {PILOT_SAMPLE_SHA256}")
        units["h1"] = [{"corpus": CORPUS[u["corpus"]], "id": u["id"], "ru": u["ru"]} for u in json.loads(raw)]
        sources["h1"] = {"file": str(PILOT / "sample.json"), "sha256": digest, "units": len(units["h1"])}
    need = [p for p in ("h2", "h3") if p in parts]
    if need:
        if not sample61.exists():
            if strict:
                sys.exit(f"{sample61} not drawn yet: run kit/sample61.py --part all --items … first")
            for p in need:
                sources[p] = {"file": str(sample61), "sha256": None, "units": None, "note": "not drawn yet"}
        else:
            raw = sample61.read_bytes()
            s = json.loads(raw)
            for p in need:
                if p not in s:
                    if strict:
                        sys.exit(f"{sample61} has no {p} draw")
                    sources[p] = {"file": str(sample61), "sha256": sha256_bytes(raw), "units": None,
                                  "note": f"no {p} draw in the file"}
                    continue
                units[p] = s[p]
                sources[p] = {"file": str(sample61), "sha256": sha256_bytes(raw), "units": len(s[p])}
                if p == "h3":
                    sources[p]["items_sha256"] = s.get("items_sha256")
    return units, sources


def take(units, n, key):
    """--limit: the first n units of a part, round-robin over its corpora (h1, h2) or
    categories (h3), so that --limit 2 reaches both translation contracts."""
    if not n:
        return units
    groups = {}
    for u in units:
        groups.setdefault(u[key], []).append(u)
    out, i = [], 0
    while len(out) < n and any(i < len(g) for g in groups.values()):
        for g in groups.values():
            if i < len(g) and len(out) < n:
                out.append(g[i])
        i += 1
    return out


# ── the request record ───────────────────────────────────────────────────────

class Capture(urllib.request.BaseHandler):
    """Sees every /api/chat request of this process, the pilot's translate.py's included.
    It states the registered options (adds num_predict, checks num_ctx and temperature) and
    records the options sent and the response's counters, call by call."""
    handler_order = 100  # before urllib's HTTPHandler fixes Content-Length

    def __init__(self):
        self.calls = []

    def http_request(self, req):
        if req.get_method() == "POST" and req.full_url.endswith("/api/chat") and req.data:
            body = json.loads(req.data)
            opts = body.setdefault("options", {})
            for k, v in OPTIONS.items():
                if k in opts and opts[k] != v:
                    raise ValueError(f"request option {k}={opts[k]!r}, registered {v!r}")
                opts[k] = v
            req.data = json.dumps(body).encode("utf-8")
            self.calls.append({"options": dict(opts), "think": body.get("think"),
                               "keep_alive": body.get("keep_alive")})
        return req

    def http_response(self, req, resp):
        if not (req.full_url.endswith("/api/chat") and self.calls and "status" not in self.calls[-1]):
            return resp
        body = resp.read()
        call = self.calls[-1]
        call["status"] = resp.code
        try:
            d = json.loads(body.decode("utf-8"))
            for k in COUNTERS:
                call[k] = d.get(k)
        except ValueError:
            call["unparsed_bytes"] = len(body)
        new = urllib.response.addinfourl(io.BytesIO(body), resp.headers, resp.url, resp.code)
        new.msg = resp.msg
        return new


def model_row(name, tag_digest):
    show = post("/api/show", {"model": name}, timeout=60)
    frm = next((ln.split(None, 1)[1].strip() for ln in show.get("modelfile", "").splitlines()
                if ln.startswith("FROM ")), None)
    row = {"name": name, "tag_digest": tag_digest, "from": frm, "blob_sha256_name": None,
           "blob_sha256_bytes": None, "details": show.get("details"),
           "parameters": show.get("parameters"), "template_sha256":
           sha256_bytes(show.get("template", "").encode("utf-8")) if show.get("template") else None,
           "architecture": (show.get("model_info") or {}).get("general.architecture"),
           "modified_at": show.get("modified_at")}
    if frm and Path(frm).name.startswith("sha256-"):
        row["blob_sha256_name"] = Path(frm).name[len("sha256-"):]
        try:
            log(f"  hashing {Path(frm).name[:23]}… for {name}")
            with open(frm, "rb") as f:
                row["blob_sha256_bytes"] = hashlib.file_digest(f, "sha256").hexdigest()
        except OSError as e:
            row["blob_sha256_bytes"] = f"unreadable: {e}"
    return row


def ps_row(model):
    for m in get("/api/ps").get("models", []):
        if m.get("name") == model or m.get("model") == model:
            return {k: m.get(k) for k in ("runner", "context_length", "size", "size_vram", "details")}
    return None


def translate_row(pr, ctx, capture, part, arm, model, u):
    capture.calls.clear()
    t0 = time.time()
    en, err, attempts, elapsed = pr.T.translate_unit(ctx, u["ru"])
    if err and err.startswith("connection:"):  # translate_unit's three attempts could not reach the daemon
        raise RunAborted(f"{arm} {part} {u['id']}: {err}")
    calls = [dict(c) for c in capture.calls]
    last = calls[-1] if calls else {}
    return {"part": part, "arm": arm, "model": model, "corpus": u["corpus"], "id": u["id"],
            "options": last.get("options"), "think": last.get("think"), "answer": en, "error": err,
            "attempts": attempts, "remedy": getattr(ctx, "remedy", None),
            "prompt_eval_count": last.get("prompt_eval_count"), "eval_count": last.get("eval_count"),
            "done_reason": last.get("done_reason"), "wall_s": round(time.time() - t0, 1),
            "elapsed_s": round(elapsed, 1), "prompt_sha": ctx.prompt_sha, "calls": calls}


def h3_row(capture, think_ok, arm, model, item):
    """One H3 prompt, three attempts as translate_unit makes them: a connection error (after a
    2·n s pause), an HTTP 5xx, a broken or empty answer is retried; an HTTP 4xx is not. A
    connection error on the last attempt stops the run (RunAborted); any other failure is the
    row's `error`. A truncated answer (done_reason "length") is kept as the model gave it."""
    capture.calls.clear()
    payload = {"model": model, "messages": [{"role": "user", "content": item["prompt"]}],
               "stream": False, "options": dict(OPTIONS)}
    if think_ok:
        payload["think"] = THINK
    t0 = time.time()
    answer, err, attempts = None, None, 0
    for attempt in range(3):
        attempts = attempt + 1
        try:
            d = post("/api/chat", payload, timeout=1800)
            answer = d["message"]["content"]
            if not answer.strip():
                err = "contract: empty answer"
                continue
            err = None
            break
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
            if e.code < 500:
                break
        except urllib.error.URLError as e:
            err = f"connection: {e.reason}"
            time.sleep(2 * (attempt + 1))
        except (OSError, http.client.HTTPException) as e:  # dropped mid-request: RemoteDisconnected, reset, read timeout
            err = f"connection: {type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
        except (ValueError, KeyError) as e:
            err = f"contract: {e}"
    if err and err.startswith("connection:"):
        raise RunAborted(f"{arm} h3 {item['id']}: {err}")
    calls = [dict(c) for c in capture.calls]
    last = calls[-1] if calls else {}
    return {"part": "h3", "arm": arm, "model": model, "category": item["category"], "id": item["id"],
            "options": last.get("options"), "think": last.get("think"), "answer": answer, "error": err,
            "attempts": attempts, "prompt_eval_count": last.get("prompt_eval_count"),
            "eval_count": last.get("eval_count"), "done_reason": last.get("done_reason"),
            "wall_s": round(time.time() - t0, 1), "calls": calls}


# ── main ─────────────────────────────────────────────────────────────────────

def write_meta(out, meta):
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def dry_run(out, meta, parts):
    made = []
    try:
        meta["ollama"] = get("/api/version", timeout=3).get("version")
        made.append("GET /api/version")
        tags = {m["name"]: m.get("digest") for m in get("/api/tags", timeout=3).get("models", [])}
        made.append("GET /api/tags")
        meta["models"] = {a: {"name": n, "tag_digest": tags.get(n), "pulled": n in tags} for a, n in ARMS.items()}
    except (urllib.error.URLError, OSError, ValueError) as e:
        meta["ollama_unreachable"] = f"{HOST}: {getattr(e, 'reason', e)}"
    meta["requests_made"] = made
    try:
        meta["contracts"] = contracts_row(pilot_run())
        meta["contracts"]["frozen_mismatch"] = frozen_mismatch(meta["contracts"])
    except Exception as e:  # recorded: the dry run proves what loads, not that everything does
        meta["contracts"] = {"error": f"{type(e).__name__}: {e}"}
    meta["thermal"]["end"] = thermal()
    meta["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_meta(out, meta)
    (out / "results.jsonl").write_text("", encoding="utf-8")
    counts = ", ".join("%s %s" % (p, meta["units"][p]["units"]) for p in parts)
    log(f"dry run → {out}: no model request; requests made: {made or 'none'}; units {counts}")


def read_done(out):
    """--resume: the (part, arm, id) of every row results.jsonl already holds. A torn last line
    or a key twice stops the resume: the record is not edited by this script."""
    f = out / "results.jsonl"
    done = set()
    if not f.exists():
        return done
    text = f.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        sys.exit(f"{f}: the last line is torn (no newline); look at it before resuming")
    for n, line in enumerate(text.splitlines(), 1):
        try:
            r = json.loads(line)
        except ValueError:
            sys.exit(f"{f} line {n} is not JSON; look at it before resuming")
        k = (r["part"], r["arm"], r["id"])
        if k in done:
            sys.exit(f"{f}: {k} twice")
        done.add(k)
    return done


def real_run(out, meta, parts, arms, units, done=None):
    """A fresh run (done is None) fills meta itself; a resumed one appends a session record to
    meta["resumes"] and skips the rows results.jsonl holds, loading only arms with work left.
    Before any request: the pilot's contracts must be the registration's frozen ones."""
    resume = done is not None
    done = done or set()
    pr = pilot_run()
    contracts = contracts_row(pr)
    bad = frozen_mismatch(contracts)
    if bad:
        sys.exit("the pilot's contracts are not the registration's frozen inputs, no model loaded: "
                 + "; ".join(bad))
    if resume and contracts != meta.get("contracts"):
        sys.exit("the pilot's contracts differ from the ones this run started with; not resumed")
    session = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "thermal": {"start": thermal(), "end": None},
               "skipped_rows": len(done)} if resume else meta
    session["contracts"] = contracts
    pk_system = pr.petukhov_system()
    capture = Capture()
    urllib.request.install_opener(urllib.request.build_opener(capture))
    session["ollama"] = get("/api/version")["version"]
    if resume and session["ollama"] != meta.get("ollama"):
        sys.exit(f"Ollama is {session['ollama']}, the run started on {meta.get('ollama')}; not resumed")
    tags = {m["name"]: m.get("digest") for m in get("/api/tags")["models"]}
    missing = [ARMS[a] for a in arms if ARMS[a] not in tags]
    if missing:
        sys.exit(f"not pulled: {missing}; the pull commands are in kit/README.md")
    todo_arms = [a for a in arms if any((p, a, u["id"]) not in done for p in parts for u in units[p])]
    session["models"] = {a: model_row(ARMS[a], tags[ARMS[a]]) for a in todo_arms}
    if resume:
        for a in todo_arms:
            was = (meta.get("models") or {}).get(a, {}).get("blob_sha256_name")
            if session["models"][a]["blob_sha256_name"] != was:
                sys.exit(f"{a}: the blob is {session['models'][a]['blob_sha256_name']}, the run started "
                         f"with {was}; not resumed")
        meta.setdefault("resumes", []).append(session)
    suffix = f"-r{len(meta['resumes'])}" if resume else ""
    (out / f"ps_before{suffix}.json").write_text(json.dumps(get("/api/ps"), indent=1) + "\n", encoding="utf-8")
    session["serving"], session["arm_seconds"], session["arms_skipped"] = {}, {}, []
    write_meta(out, meta)
    log(f"run {out.name}{' (resumed, ' + str(len(done)) + ' rows kept)' if resume else ''}: arms {todo_arms}, "
        f"parts {parts}, units " + ", ".join(f"{p} {len(units[p])}" for p in parts) + f", contracts {contracts}")
    try:
        with (out / "results.jsonl").open("a" if resume else "w", encoding="utf-8") as fh:
            for arm in todo_arms:
                model = ARMS[arm]
                ctxs = pr.contexts(model, pk_system)
                for c in ctxs.values():
                    c.host = HOST
                if not pr.T.warmup(ctxs["mm"]):
                    warn(f"{arm}: {model} not usable; arm skipped")
                    session["arms_skipped"].append(arm)
                    continue
                session["serving"][arm] = dict(SERVING_ENV, ps=ps_row(model), think_accepted=ctxs["mm"].caps["think"])
                t_arm = time.time()
                for part in parts:
                    todo = [u for u in units[part] if (part, arm, u["id"]) not in done]
                    for n, u in enumerate(todo, 1):
                        if part == "h3":
                            row = h3_row(capture, ctxs["mm"].caps["think"], arm, model, u)
                        else:
                            row = translate_row(pr, ctxs[PILOT_CODE[u["corpus"]]], capture, part, arm, model, u)
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                        if row["error"]:
                            warn(f"{arm} {part} {u['id']}: {row['error']}")
                        if n % 5 == 0 or n == len(todo):
                            log(f"  {arm} {part} [{n}/{len(todo)}] {row['wall_s']} s last unit")
                session["arm_seconds"][arm] = round(time.time() - t_arm, 1)
                post("/api/generate", {"model": model, "keep_alive": 0}, timeout=120)
                log(f"  {arm}: {model} unloaded (keep_alive 0)")
    finally:
        try:
            post("/api/generate", {"model": BOT_MODEL, "keep_alive": -1}, timeout=600)
            session["bot_reloaded"] = True
            log(f"{BOT_MODEL} reloaded with keep_alive -1")
        except Exception as e:  # the record says so; the operator restores it by hand
            session["bot_reloaded"] = f"failed: {e}"
            warn(f"could not reload {BOT_MODEL}: {e}")
        try:
            (out / f"ps_after{suffix}.json").write_text(json.dumps(get("/api/ps"), indent=1) + "\n", encoding="utf-8")
        except Exception as e:
            warn(f"ps_after not written: {e}")
        session["thermal"]["end"] = thermal()
        session["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_meta(out, meta)
        log(f"run {out.name} ended → {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", default="h1,h2,h3")
    ap.add_argument("--arms", default="Q,A")
    ap.add_argument("--limit", type=int, default=0, help="units per part per arm (the smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="meta.json and an empty results.jsonl; no model request")
    ap.add_argument("--sample61", type=Path, default=SAMPLE61)
    ap.add_argument("--out", type=Path, help="run folder (default runs/<UTC stamp>[-smoke|-dry])")
    ap.add_argument("--resume", type=Path, help="a stopped run's folder: its parts, arms, limit and units, "
                                                "results.jsonl appended, rows it holds skipped")
    args = ap.parse_args()

    if args.resume:
        if args.out or args.dry_run or args.parts != "h1,h2,h3" or args.arms != "Q,A" or args.limit \
                or args.sample61 != SAMPLE61:
            ap.error("--resume takes the folder's own parts, arms, limit and sample; give it alone")
        out = args.resume if args.resume.is_absolute() else Path.cwd() / args.resume
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        if meta.get("dry_run"):
            sys.exit(f"{out} is a dry run: nothing to resume")
        parts, arms, limit = meta["parts"], list(meta["arms"]), meta["limit"]
        s61 = next((Path(meta["units"][p]["file"]) for p in ("h2", "h3") if p in meta["units"]), SAMPLE61)
        units, sources = load_units(parts, s61, strict=True)
        for p in parts:
            if sources[p]["sha256"] != meta["units"][p]["sha256"]:
                sys.exit(f"{sources[p]['file']} changed since the run started; not resumed")
            units[p] = take(units[p], limit, "category" if p == "h3" else "corpus")
        done = read_done(out)
    else:
        parts = [p.strip() for p in args.parts.split(",") if p.strip()]
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        if not parts or any(p not in PARTS for p in parts) or not arms or any(a not in ARMS for a in arms):
            ap.error(f"parts from {PARTS}, arms from {tuple(ARMS)}")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        tag = "-dry" if args.dry_run else ("-smoke" if args.limit else "")
        out = args.out or EXP / "runs" / (stamp + tag)
        if (out / "meta.json").exists():
            sys.exit(f"{out} already holds a run (to continue a stopped one: --resume {out})")
        out.mkdir(parents=True, exist_ok=True)
        units, sources = load_units(parts, args.sample61, strict=not args.dry_run)
        for p in units:
            units[p] = take(units[p], args.limit, "category" if p == "h3" else "corpus")
        for p in sources:
            if p in units:
                sources[p]["units_run"] = len(units[p])
        meta = {"stamp": out.name, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dry_run": bool(args.dry_run),
                "host": HOST, "seed": SEED, "arms": {a: ARMS[a] for a in arms}, "parts": parts,
                "limit": args.limit, "num_ctx": NUM_CTX, "temperature": TEMPERATURE, "think": THINK,
                "num_predict": NUM_PREDICT, "keep_alive_after_arm": 0,
                "bot_model": {"name": BOT_MODEL, "keep_alive": -1}, "ollama": None, "models": None,
                "serving": None, "serving_env": SERVING_ENV, "os": os_row(),
                "thermal": {"start": thermal(), "end": None}, "units": sources}
        done = None
        if args.dry_run:
            return dry_run(out, meta, parts)
    try:
        real_run(out, meta, parts, arms, units, done)
    except RunAborted as e:
        sys.exit(f"stopped: {e}. The unit has no row; continue with: python3.14 -B kit/run61.py --resume {out}")
    except (OSError, http.client.HTTPException) as e:  # a socket dropped inside translate_unit, which retries URLError only
        sys.exit(f"stopped: {type(e).__name__}: {e}. The unit has no row; continue with: "
                 f"python3.14 -B kit/run61.py --resume {out}")
    except KeyboardInterrupt:
        sys.exit(f"interrupted; continue with: python3.14 -B kit/run61.py --resume {out}")


if __name__ == "__main__":
    main()
