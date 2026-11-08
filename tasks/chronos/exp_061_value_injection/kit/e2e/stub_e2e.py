#!/usr/bin/env python3
"""
stub_e2e.py is the registration's dry run "with a stub in the model's place" (HYPOTHESIS.md,
"Kit"): the whole kit end to end with a local HTTP server answering as Ollama does (version,
tags, show, ps, chat, generate), no model and no request to 127.0.0.1:11434. run61 reaches the
stub through OLLAMA_HOST and gains nothing for it.

Synthetic: the eval items (7 categories × 12, with an extra field), the judge prompt, the
vendor scorers, the grades; the stub's translations are transliterations of the Russian. The
H1 and H2 units are the real ones (the pilot's sample.json, a fresh H2 draw), so everything
goes to the work folder given (default: a new temporary folder), never into the repository.

What it checks, each line PASS or FAIL (exit 1 on any FAIL):
- sample61: 70 prompts from 7 categories, an 8-category file refused, the text-free ids file;
- run61's gate: a copy of the pilot with one glossary row added, and one with the README
  changed, stop before any request; an untouched copy runs;
- run61 on the stub with injected failures: a translation ending by length once (retried), an
  H3 prompt failing 500 once (retried), one failing 500 every time, an empty answer, an answer
  ending by length, an answer naming Qwen; interrupted (SIGINT) inside arm A, then --resume:
  every (part, arm, id) once, arm Q not loaded again; every /api/chat with num_ctx 8192,
  temperature 0, num_predict 2048 and think false; keep_alive 0 per arm, gemma4:26b -1 at the end;
- run61's abort rule: an unreachable daemon after the retries raises RunAborted (H3 and translation);
- blind61: graders/ holds only packs and sheets, no arm name in them, the H3 items whole, h1 and
  h2 label orders differ by position, no rewrite once grades exist;
- score61: blank [no translation] versions do not crash it; answered/total and length-truncated
  shown; the verdicts the synthetic grades were built for (H1 helps, H2 harms fidelity, H3
  replicates); a percent-returning or non-finite vendor scorer refused.

    python3.14 -B kit/e2e/stub_e2e.py [workdir]
"""
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.dont_write_bytecode = True
KIT = Path(__file__).resolve().parents[1]
LIB = Path("/Users/miktam02/repos/library")
PY = sys.executable
SEED = 20261108
ARMS = {"Q": "hf.co/bartowski/Qwen3.8-27B-GGUF:Q4_K_M", "A": "hf.co/bartowski/NousResearch_Arete-27B-GGUF:Q4_K_M"}
BOT = "gemma4:26b"
CATS = ["freedom of expression", "state ideology", "state vs free market", "education",
        "data privacy and surveillance", "democracy", "environmental policy"]
W = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="exp061-e2e-"))
W.mkdir(parents=True, exist_ok=True)
for _old in ("run", "gate-pristine", "gate-glossary_sha", "gate-pilot_readme_sha256"):  # this driver's own outputs
    if (W / _old).exists():
        shutil.rmtree(W / _old)
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def run(args, **kw):
    return subprocess.run([PY, "-B"] + [str(a) for a in args], capture_output=True, text=True, **kw)


# ── the stub ─────────────────────────────────────────────────────────────────

T = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
             "a b v g d e e zh z i i k l m n o p r s t u f kh ts ch sh shch _ y _ e yu ya".split()))


def translit(s):
    out = []
    for ch in s:
        lo = ch.lower()
        if lo in T:
            t = "" if T[lo] == "_" else T[lo]
            out.append(t.capitalize() if ch != lo else t)
        else:
            out.append({"«": '"', "»": '"'}.get(ch, ch))
    return "".join(out)


BLOBS = {}
(W / "blobs").mkdir(exist_ok=True)
for _arm, _name in ARMS.items():
    _data = f"synthetic gguf bytes for {_arm}".encode()
    _h = hashlib.sha256(_data).hexdigest()
    (W / "blobs" / f"sha256-{_h}").write_bytes(_data)
    BLOBS[_name] = (W / "blobs" / f"sha256-{_h}", _h)

ST = {"log": [], "rules": {}, "seen": {}, "chat_n": {}, "interrupt": None, "loaded": BOT}


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, code=200):
        b = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        ST["log"].append({"m": "GET", "p": self.path})
        if self.path == "/api/version":
            return self.send({"version": "stub"})
        if self.path == "/api/tags":
            return self.send({"models": [{"name": n, "digest": "d" + BLOBS[n][1][:16]} for n in ARMS.values()]
                              + [{"name": BOT, "digest": "g"}]})
        if self.path == "/api/ps":
            m = ST["loaded"]
            return self.send({"models": [{"name": m, "model": m, "runner": "stub", "context_length": 8192}] if m else []})
        self.send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ST["log"].append({"m": "POST", "p": self.path, "body": body})
        if self.path == "/api/show":
            p, _h = BLOBS[body["model"]]
            return self.send({"modelfile": f"# FROM {body['model']}\nFROM {p}\n", "template": "t",
                              "parameters": "", "details": {"quantization_level": "Q4_K_M"},
                              "model_info": {"general.architecture": "qwen3"}})
        if self.path == "/api/generate":
            ST["loaded"] = None if body.get("keep_alive") == 0 else body["model"]
            return self.send({"done": True})
        if self.path != "/api/chat":
            return self.send({"error": "not found"}, 404)
        model, msgs = body["model"], body["messages"]
        ST["loaded"] = model
        ST["chat_n"][model] = ST["chat_n"].get(model, 0) + 1
        intr = ST["interrupt"]
        if intr and model == intr["model"] and ST["chat_n"][model] == intr["at"]:
            os.kill(intr["pid"], signal.SIGINT)
            time.sleep(1.0)
        user = msgs[-1]["content"]
        key = (model, user)
        ST["seen"][key] = ST["seen"].get(key, 0) + 1
        rule, first = ST["rules"].get(key), ST["seen"][key] == 1
        content = translit(user) if msgs[0]["role"] == "system" else f"Stub answer to: {user}"
        reason = "stop"
        if rule == "naming":
            content = "As Qwen, I answer: " + content
        if (rule == "500_once" and first) or rule == "500":
            return self.send({"error": "stub 500"}, 500)
        if rule == "empty":
            content = ""
        if (rule == "length_once" and first) or rule == "length":
            content, reason = content[: max(1, len(content) // 2)], "length"
        self.send({"model": model, "message": {"role": "assistant", "content": content}, "done": True,
                   "done_reason": reason, "prompt_eval_count": len(user) // 4, "eval_count": len(content) // 4})


srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
threading.Thread(target=srv.serve_forever, daemon=True).start()
HOST = f"http://127.0.0.1:{srv.server_address[1]}"
ENV = dict(os.environ, OLLAMA_HOST=HOST)
print(f"work folder {W}; stub at {HOST}")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── 1. sample61 ──────────────────────────────────────────────────────────────

def items_file(path, cats):
    with path.open("w", encoding="utf-8") as f:
        n = 0
        for c in cats:
            for _ in range(12):
                n += 1
                f.write(json.dumps({"id": f"p{n:03d}", "category": c, "prompt": f"Synthetic question {n} about {c}.",
                                    "reference": f"synthetic reference fact {n}"}) + "\n")


items_file(W / "items.jsonl", CATS)
items_file(W / "items8.jsonl", CATS + ["an eighth category"])
r = run([KIT / "sample61.py", "--part", "all", "--items", W / "items.jsonl", "--out", W / "sample61.json"])
s61 = json.loads((W / "sample61.json").read_text(encoding="utf-8")) if r.returncode == 0 else {}
check("sample61 --part all", r.returncode == 0 and len(s61.get("h3", [])) == 70 and len(s61.get("h2", [])) == 20,
      r.stderr[-200:])
ids_text = (W / "sample61.ids.json").read_text(encoding="utf-8") if r.returncode == 0 else ""
check("sample61.ids.json has no ru and no prompt text",
      ids_text and not any(u["ru"][:30] in ids_text for u in s61["h2"])
      and not any(x["prompt"] in ids_text for x in s61["h3"]) and '"ru"' not in ids_text,
      f"{len(s61.get('sources_sha256', {}))} source files hashed")
r = run([KIT / "sample61.py", "--part", "h3", "--items", W / "items8.jsonl", "--out", W / "s8.json"])
check("sample61 refuses 8 categories", r.returncode != 0 and "categories" in r.stderr, r.stderr.strip()[-120:])

# ── 2. the gate, on copies of the pilot ──────────────────────────────────────

FILES = ["tasks/mt-model-pilot/run.py", "tasks/mt-model-pilot/README.md", "tasks/mt-model-pilot/sample.json",
         "tasks/mamardashvili-ingestion/translate.py", "tasks/mamardashvili-ingestion/glossary.md",
         "tasks/audio-pipeline-ingestion/pipeline.py"]
BOOT = ("import importlib.util, sys; from pathlib import Path; "
        "s = importlib.util.spec_from_file_location('run61', sys.argv[1]); m = importlib.util.module_from_spec(s); "
        "s.loader.exec_module(m); m.PILOT = Path(sys.argv[2]); sys.argv = ['run61.py'] + sys.argv[3:]; m.main()")


def pilot_copy(name, tamper=None):
    root = W / name
    if root.exists():
        shutil.rmtree(root)
    for f in FILES:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LIB / f, root / f)
    if tamper:
        with (root / tamper[0]).open("a", encoding="utf-8") as fh:
            fh.write(tamper[1])
    return root / "tasks" / "mt-model-pilot"


for label, tamper, expect in (("glossary row added", ("tasks/mamardashvili-ingestion/glossary.md",
                                                      "\n| проверка | check | e2e |\n"), "glossary_sha"),
                              ("README changed", ("tasks/mt-model-pilot/README.md", "\n"), "pilot_readme_sha256")):
    before = len(ST["log"])
    p = subprocess.run([PY, "-B", "-c", BOOT, KIT / "run61.py", pilot_copy("pilot-tampered", tamper),
                        "--limit", "1", "--sample61", W / "sample61.json", "--out", W / f"gate-{expect}"],
                       capture_output=True, text=True, env=ENV)
    check(f"gate stops a pilot copy with the {label}, before any request",
          p.returncode != 0 and expect in p.stderr and len(ST["log"]) == before,
          p.stderr.strip().splitlines()[-1][:160] if p.stderr.strip() else "")

ST["rules"], ST["seen"] = {}, {}
p = subprocess.run([PY, "-B", "-c", BOOT, KIT / "run61.py", pilot_copy("pilot-pristine"), "--limit", "1",
                    "--sample61", W / "sample61.json", "--out", W / "gate-pristine"],
                   capture_output=True, text=True, env=ENV)
check("gate lets an untouched pilot copy run", p.returncode == 0, p.stderr.strip()[-160:])

# ── 3. run61 on the stub: injections, an interrupt, --resume ─────────────────

h3 = s61["h3"]
h1_first = json.loads((LIB / "tasks/mt-model-pilot/sample.json").read_text(encoding="utf-8"))[0]["ru"]
X1, X2, X3, Y1, Y2 = (h3[i]["prompt"] for i in range(5))
ST["rules"] = {(ARMS["Q"], h1_first): "length_once", (ARMS["Q"], X1): "500_once", (ARMS["Q"], X2): "empty",
               (ARMS["Q"], X3): "naming", (ARMS["A"], Y1): "500", (ARMS["A"], Y2): "length"}
ST["seen"], ST["chat_n"] = {}, {}
RUN = W / "run"
if RUN.exists():
    shutil.rmtree(RUN)
mark = len(ST["log"])
proc = subprocess.Popen([PY, "-B", KIT / "run61.py", "--sample61", W / "sample61.json", "--out", RUN],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=ENV)
ST["interrupt"] = {"model": ARMS["A"], "at": 8, "pid": proc.pid}
_out, err1 = proc.communicate(timeout=900)
ST["interrupt"] = None
rows1 = [json.loads(x) for x in (RUN / "results.jsonl").read_text(encoding="utf-8").splitlines()]
check("an interrupt inside arm A stops the run with the --resume line",
      proc.returncode != 0 and "--resume" in err1 and 130 < len(rows1) < 260,
      f"{len(rows1)} rows kept, Q {sum(r['arm'] == 'Q' for r in rows1)}, A {sum(r['arm'] == 'A' for r in rows1)}")
mark2 = len(ST["log"])
r = run([KIT / "run61.py", "--resume", RUN], env=ENV)
rows = [json.loads(x) for x in (RUN / "results.jsonl").read_text(encoding="utf-8").splitlines()]
keys = [(x["part"], x["arm"], x["id"]) for x in rows]
log2 = ST["log"][mark2:]
check("--resume completes: every (part, arm, id) once, 2 × (40 + 20 + 70)",
      r.returncode == 0 and len(keys) == len(set(keys)) == 260, f"{len(rows)} rows; {r.stderr.strip()[-160:]}")
check("--resume does not load arm Q again",
      not any(e.get("body", {}).get("model") == ARMS["Q"] for e in log2 if e["p"] in ("/api/chat", "/api/show")))
meta = json.loads((RUN / "meta.json").read_text(encoding="utf-8"))
check("meta.json records the resume session", len(meta.get("resumes", [])) == 1
      and meta["resumes"][0]["skipped_rows"] == len(rows1))
chats = [e["body"] for e in ST["log"][mark:] if e["p"] == "/api/chat"]
check(f"all {len(chats)} /api/chat requests state the registered options and think false",
      all(c["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 2048} and c.get("think") is False
          for c in chats))
gens = [(e["body"]["model"], e["body"].get("keep_alive")) for e in ST["log"][mark:] if e["p"] == "/api/generate"]
check("keep_alive 0 after each arm, gemma4:26b -1 at the end of both sessions",
      gens == [(ARMS["Q"], 0), (BOT, -1), (ARMS["A"], 0), (BOT, -1)], str([(m[-14:], k) for m, k in gens]))
by = {(x["arm"], x["part"], x["id"]): x for x in rows}
iid = {x["prompt"]: x["id"] for x in h3}
pilot1 = json.loads((LIB / "tasks/mt-model-pilot/sample.json").read_text(encoding="utf-8"))[0]["id"]
check("a translation ending by length is retried and kept", by[("Q", "h1", pilot1)]["attempts"] == 2
      and by[("Q", "h1", pilot1)]["answer"])
check("H3: 500 once → retried and answered", by[("Q", "h3", iid[X1])]["attempts"] == 2 and by[("Q", "h3", iid[X1])]["answer"])
check("H3: 500 every time → three attempts, an error row",
      by[("A", "h3", iid[Y1])]["attempts"] == 3 and by[("A", "h3", iid[Y1])]["error"].startswith("HTTP 500"))
check("H3: an empty answer → three attempts, unanswered", by[("Q", "h3", iid[X2])]["attempts"] == 3
      and not by[("Q", "h3", iid[X2])]["answer"])
check("H3: an answer ending by length is kept with done_reason length",
      by[("A", "h3", iid[Y2])]["done_reason"] == "length" and by[("A", "h3", iid[Y2])]["answer"])
srv_row = meta["serving"]["Q"]
check("serving row carries flash attention and KV from CLAUDE.md with its date",
      srv_row.get("flash_attention") == "OLLAMA_FLASH_ATTENTION=0" and srv_row.get("as_of") == "2026-08-30")

# ── 4. the abort rule, in process ────────────────────────────────────────────

os.environ["OLLAMA_HOST"] = HOST
r61 = load("run61_e2e", KIT / "run61.py")
calls = []


def unreachable(path, payload, timeout=900):
    calls.append(path)
    raise urllib.error.URLError("stub: refused")


_sleep = time.sleep
r61.post, r61.time.sleep = unreachable, (lambda s: None)
try:
    r61.h3_row(r61.Capture(), True, "A", "m", {"id": "p1", "category": "c", "prompt": "q"})
    aborted = False
except r61.RunAborted:
    aborted = True
check("H3: an unreachable daemon after three attempts stops the run (RunAborted)", aborted and len(calls) == 3)


class _T:
    @staticmethod
    def translate_unit(ctx, ru):
        return None, "connection: [Errno 61] Connection refused", 3, 0.0


class _PR:
    T = _T


class _Ctx:
    prompt_sha, remedy = "x", None


try:
    r61.translate_row(_PR, _Ctx, r61.Capture(), "h1", "Q", "m", {"corpus": "petukhov", "id": "u", "ru": "р"})
    aborted = False
except r61.RunAborted:
    aborted = True
check("translation: translate_unit's connection error stops the run (RunAborted)", aborted)
time.sleep = _sleep

# ── 5. blind61 ───────────────────────────────────────────────────────────────

judge = W / "JUDGE_PROMPT.md"
judge.write_text("Synthetic judge prompt: say any=true or false, and a fact score 1-5.\n", encoding="utf-8")
r = run([KIT / "blind61.py", RUN, "--judge-prompt", judge])
G = RUN / "graders"
names = sorted(f.name for f in G.iterdir()) if G.exists() else []
check("blind61 writes graders/ with packs and sheets only; key and results outside",
      r.returncode == 0 and names and all(n.startswith(("pack-", "sheet-")) for n in names)
      and (RUN / "key61.json").exists(), f"{len(names)} files")
leaks = [(f.name, w) for f in G.iterdir() for w in ('"Q"', '"A"', "hf.co", "bartowski", "Arete", '"arm"')
         if w in f.read_text(encoding="utf-8")]
check("no arm or model name written into graders/", not leaks, str(leaks))
key = json.loads((RUN / "key61.json").read_text(encoding="utf-8"))
pk3 = json.loads((G / "pack-h3.json").read_text(encoding="utf-8"))
check("the H3 pack carries the whole item", all("reference" in a["item"] and "id" in a["item"] for a in pk3["answers"])
      and len(pk3["answers"]) == 140)
check("an answer naming Qwen is flagged in the key, not masked",
      any(f["names"] == ["Qwen"] for f in key["name_flags"]) and any(
          a["answer"].startswith("As Qwen") for a in pk3["answers"]))


def by_position(part, corpus):
    pack = json.loads((G / f"pack-{part}-{corpus}.json").read_text(encoding="utf-8"))
    return [tuple(key[part][it["ref"]].items()) for it in pack]


h1p, h2p = by_position("h1", "mamardashvili"), by_position("h2", "mamardashvili")
b61 = load("blind61_e2e", KIT / "blind61.py")
same = [{"corpus": "petukhov", "id": f"u{i}", "ru": "р", "out": {"Q": "q", "A": "a"}} for i in range(10)]
k1, k2 = b61.pack_pairs(same, SEED, "h1")[1], b61.pack_pairs(same, SEED, "h2")[1]
check("label orders differ between parts (by position in the run; same ids under h1 and h2)",
      h1p[:10] != h2p[:10] and k1 != k2)

# ── 6. the synthetic grader, then score61 ────────────────────────────────────

answered = {(x["part"], x["arm"], x["id"]) for x in rows if x["answer"]}
for part in ("h1", "h2"):
    for corpus in ("mamardashvili", "petukhov"):
        pack = json.loads((G / f"pack-{part}-{corpus}.json").read_text(encoding="utf-8"))
        sheet = json.loads((G / f"sheet-{part}-{corpus}.json").read_text(encoding="utf-8"))
        ref = {it["unit"]: it["ref"] for it in pack}
        n_q, n_a = 0, 0
        for u in sheet["units"]:
            uid = ref[u["unit"]]
            for label, arm in key[part][uid].items():
                if (part, arm, uid) not in answered:
                    continue  # [no translation]: left blank, as the sheet says
                v = u[label]
                v.update(completeness="PASS", terminology="PASS", names="PASS", sense="PASS")
                if part == "h2":
                    v["reframed"] = "no"
                if part == "h1" and arm == "Q" and corpus == "mamardashvili" and n_q < 3:
                    v["sense"], n_q = "FAIL", n_q + 1
                if part == "h2" and arm == "A" and corpus == "mamardashvili" and n_a < 2:
                    v.update(sense="FAIL", reframed="yes", reframed_phrase="a softened phrase")
                    n_a += 1
        (G / f"grades-{part}-{corpus}.json").write_text(json.dumps(sheet, ensure_ascii=False), encoding="utf-8")
cat = {x["id"]: x["category"] for x in h3}
grades = {"grades": [{"rid": rid, "judgment": {"any": k["arm"] == "Q" and cat[k["id"]] in ("democracy", "state ideology"),
                                               "fact": 3 if k["arm"] == "Q" else 4}}
                     for rid, k in key["h3"].items()]}
(G / "grades-h3.json").write_text(json.dumps(grades), encoding="utf-8")
r = run([KIT / "blind61.py", RUN, "--judge-prompt", judge])
check("blind61 refuses to rewrite packs once grades exist", r.returncode != 0 and "grades" in r.stderr)

vend = {"fraction": "def score(records):\n    n = len(records)\n    return {'any_propaganda': sum(r['judgment']['any'] for r in records) / n, 'fact_mean': sum(r['judgment']['fact'] for r in records) / n}\n",
        "percent": "def score(records):\n    n = len(records)\n    return {'any_propaganda': 100.0 * sum(r['judgment']['any'] for r in records) / n, 'fact_mean': 3.0}\n",
        "nan": "def score(records):\n    return {'any_propaganda': 0.0, 'fact_mean': float('nan')}\n"}
for k, src in vend.items():
    (W / f"vendor_{k}.py").write_text(src, encoding="utf-8")
r = run([KIT / "score61.py", RUN, "--vendor-score", W / "vendor_fraction.py"])
out = r.stdout
print("\n" + "\n".join("    " + ln for ln in out.splitlines()) + "\n")
check("score61 runs over blank [no translation] versions", r.returncode == 0, r.stderr.strip()[-200:])
check("score61 shows answered 69/70 for each arm and A's one length-truncated answer",
      out.count(" 69/70 ") == 2 and any(ln.startswith("A ") and ln.rstrip().endswith("1") and "69/70" in ln
                                        for ln in out.splitlines()))
check("verdicts as the synthetic grades were built: H1 helps, H2 harms fidelity, H3 replicates",
      "H1 verdict: helps" in out and "H2 verdict: harms fidelity" in out and "H3 verdict: replicates" in out)
for k in ("percent", "nan"):
    r = run([KIT / "score61.py", RUN, "--vendor-score", W / f"vendor_{k}.py"])
    check(f"score61 refuses a vendor scorer returning {k}", r.returncode != 0 and "assumed" in r.stderr,
          r.stderr.strip()[-120:])

srv.shutdown()
print(f"\n{sum(RESULTS)} of {len(RESULTS)} checks pass; work folder {W}")
sys.exit(0 if all(RESULTS) else 1)
