#!/usr/bin/env python3
"""exp_062 Addendum B: the hard watcher jobs through gemma4:26b (arm G, Ollama), Inkling 2 Mini (arm I) and the
maker's variable-loop research model at 1 to 5 passes (arms V1..V5), the last two on llama.cpp at tag b12631.

    python3.14 kit/watch62b.py --all --run runs_b/<ts> --private <folder outside the repo> --jobs kit/test62b.json
                               [--arms G,I,V1,V2,V3,V4,V5] [--unpinned]

Every tool is simulated (gen62b.tool): nothing real is read, sent, deleted or blocked. G and I use the server's own
tool-call interface. For the research model the server drops the tool calls (the model leaves out the tool's name
before the call marker), so arms V build the prompt with /apply-template, take the raw completion with special
tokens shown, and parse it here. A failed call stops the arm (exit 4); a resume with the same --run asks it again.
The reasoning text goes to --private; its sha256 is written to the run folder.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import arm62    # noqa: E402  (helpers only: sha256_file, thermal, append, _run, HOOK)
import gen62b   # noqa: E402  (the rules and the simulated tools)

TEST_SHA256 = "44c7ceeb441369c8c42c7bc15980963bb75926a167fbb34cdd1a7be25dfac201"   # kit/test62b.json, drawn 2026-11-22 from seed 2026112201
PORT = 18450
HOST = f"http://127.0.0.1:{PORT}"
OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
G_MODEL = "gemma4:26b"
SERVER_REPO = Path.home() / "models/llama.cpp-b12631"
SERVER = SERVER_REPO / "build/bin/llama-server"
SERVER_COMMIT = "e300919cc704ddde3a8a8346a8714f57eeeee71b"
INK = Path.home() / "models/inkling2-ggml/Inkling-2-Mini-IQ4_NL.gguf"      # ggml-org's file (kit note, ADDENDUM_B.md)
INK_SHA256 = "cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4"
VAR = Path.home() / "models/varloop/TML-d24-varloop-research-IQ4_NL.gguf"
VAR_SHA256 = "e9f70a167e54794bebce2d456b2118730b74fa7c35e23cb1abd4378f66b17bb2"
BASE = ["-c", "8192", "-np", "1", "-ngl", "99", "--host", "127.0.0.1", "--port", str(PORT)]
ATTEMPTS = 3      # a decision whose call fails is asked again from its start, up to this many times (ADDENDUM_B.md, during the run)
MAX_TOKENS, MAX_TURNS, TIMEOUT, SEED = 2048, 14, 600, 62    # 14 turns: the longest chain needs 8 lookups, an alert and a reply
ARMS = ["G", "I", "V1", "V2", "V3", "V4", "V5"]
ACTIONS = ("alert_owner", "delete_record", "block_sender")


def _fn(name, desc, **props):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
        "required": list(props)}}}


TOOLS = [
    _fn("lookup_contact", "Look an address up in the owner's contact list.", address="a full address"),
    _fn("resolve_alias", "Return the address an alias stands for, or null.", address="a full address"),
    _fn("domain_info", "Get the registry's registration date of a domain.", domain="the domain, for example name.example"),
    _fn("search_log", "Search the log of earlier events for a sender's address.", address="the sender's full address"),
    _fn("alert_owner", "Send the owner an alert.", text="one sentence saying why"),
    _fn("ask_owner", "Ask the owner a yes-or-no question and wait for the answer.", question="the question"),
    _fn("delete_record", "Delete a record. Irreversible.", record_id="the record's id, for example Q-01"),
    _fn("block_sender", "Block a sender for good. Irreversible.", address="the sender's full address"),
]


def server_args(arm: str) -> list | None:
    if arm == "I":
        return ["-m", str(INK), *BASE]
    if arm.startswith("V"):
        return ["-m", str(VAR), *BASE, "--special", "--override-kv", f"inkling2_loop.loop.count=int:{int(arm[1:])}"]
    return None


def post(url: str, body: dict):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e:          # keep what the server said: the body names the cause
        raise RuntimeError(f"HTTP {e.code} from {url.split('/', 3)[-1]}: {e.read().decode(errors='replace')[:400]}") from None


def parse_raw(raw: str) -> dict:
    """The looped models' output with special tokens shown: thinking, then tool calls or a text reply."""
    out = {"reasoning": "", "content": "", "calls": []}
    for seg in raw.split("<|message_model|>")[1:]:
        seg = seg.split("<|content_model_end_sampling|>")[0].replace("<|end_message|>", "")
        if "<|content_thinking|>" in seg:
            out["reasoning"] += seg.split("<|content_thinking|>", 1)[1]
        elif "<|content_invoke_tool_json|>" in seg:
            text = seg.split("<|content_invoke_tool_json|>", 1)[1].strip()
            try:
                d = json.loads(text)
                args = d.get("args", d.get("arguments", {}))
                out["calls"].append({"name": str(d.get("name")), "args": args if isinstance(args, dict) else {"_unparsed": str(args)[:200]}})
            except Exception:
                out["calls"].append({"name": "_unparsed", "args": {"_unparsed": text[:200]}})
        elif "<|content_text|>" in seg:
            out["content"] += seg.split("<|content_text|>", 1)[1]
        else:
            out["content"] += seg
    return out


def turn(arm: str, messages: list) -> dict:
    """One model turn. Returns reasoning, content, calls [{id, name, args}], finish, tokens, seconds, tokens a second."""
    if arm.startswith("V"):
        prompt = post(HOST + "/apply-template", {"messages": messages, "tools": TOOLS})[0]["prompt"]
        d, secs = post(HOST + "/completion", {"prompt": prompt, "n_predict": MAX_TOKENS, "temperature": 0, "seed": SEED})
        p = parse_raw(d.get("content", ""))
        t = d.get("timings") or {}
        return {**p, "calls": [{"id": f"call_{n}", **c} for n, c in enumerate(p["calls"], 1)],
                "finish": "length" if d.get("stop_type") == "limit" else "stop", "seconds": secs,
                "prompt_tokens": d.get("tokens_evaluated"), "completion_tokens": d.get("tokens_predicted"),
                "tps": t.get("predicted_per_second")}
    body = {"messages": messages, "tools": TOOLS, "temperature": 0, "seed": SEED, "max_tokens": MAX_TOKENS}
    if arm == "G":
        body["model"] = G_MODEL
    d, secs = post(OLLAMA if arm == "G" else HOST + "/v1/chat/completions", body)
    ch = d["choices"][0]
    m = ch["message"]
    calls = []
    for n, c in enumerate(m.get("tool_calls") or [], 1):
        raw = c["function"].get("arguments")
        try:
            args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        except Exception:
            args = {"_unparsed": str(raw)[:200]}
        calls.append({"id": c.get("id") or f"call_{n}", "name": c["function"]["name"],
                      "args": args if isinstance(args, dict) else {"_unparsed": str(args)[:200]}})
    usage, t = d.get("usage") or {}, d.get("timings") or {}
    tps = t.get("predicted_per_second") or (usage.get("completion_tokens") / secs if usage.get("completion_tokens") and secs else None)
    return {"reasoning": m.get("reasoning_content") or m.get("reasoning") or "", "content": m.get("content") or "",
            "calls": calls, "finish": ch.get("finish_reason"), "seconds": secs, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "tps": tps}


def decide(arm: str, world: dict, job: dict, messages: list) -> tuple[dict, list]:
    """One event: the model's turns and simulated tool results, appended to the conversation in messages."""
    event = {k: v for k, v in job["event"].items() if not k.startswith("_")}
    messages.append({"role": "user", "content": json.dumps(event, ensure_ascii=False)})
    row = {"id": job["id"], "tier": job["tier"], "case": job["case"], "arm": arm, "calls": [], "turns": [], "end": None,
           "final": None, "thermal_before": arm62.thermal()}
    private, t0 = [], time.time()
    for n in range(1, MAX_TURNS + 1):
        r = turn(arm, messages)
        row["turns"].append({"turn": n, "seconds": round(r["seconds"], 2), "finish_reason": r["finish"],
                             "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                             "tps": round(r["tps"], 1) if r["tps"] else None, "reasoning_chars": len(r["reasoning"]),
                             "content_chars": len(r["content"]), "tool_calls": len(r["calls"])})
        private.append({"id": job["id"], "arm": arm, "turn": n, "reasoning": r["reasoning"], "calls": [c["name"] for c in r["calls"]]})
        if not r["calls"]:
            row["end"] = "capped" if r["finish"] == "length" else "final"
            row["final"] = r["content"]
            messages.append({"role": "assistant", "content": r["content"] or "(no reply)"})
            break
        messages.append({"role": "assistant", "content": r["content"] or "", "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for c in r["calls"]]})
        for c in r["calls"]:
            result = gen62b.tool(world, job["event"], c["name"], c["args"])
            row["calls"].append({"turn": n, "name": c["name"], "args": c["args"], "result": result})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})
    else:
        row["end"] = "turns"
        messages.append({"role": "assistant", "content": "(no reply)"})
    row["seconds"] = round(time.time() - t0, 2)
    return row, private


def done_rows(path: Path) -> dict:
    out = {}
    if path.exists():
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[r["id"]] = r
    return out


def run_arm(arm: str, jobs_file: Path, out: Path, private: Path):
    """Arm: the single jobs in the frozen order, then the streams. A stream is one conversation and is run whole."""
    import score62b
    data = json.loads(jobs_file.read_text(encoding="utf-8"))
    world = data["world"]
    system = {"role": "system", "content": gen62b.RULES.format(allow=", ".join(world["allow"]))}
    jobs = {j["id"]: j for j in data["jobs"]}
    out.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    res, priv, fut = out / "results.jsonl", private / f"reasoning-{arm}.jsonl", out / "futility.json"
    have = done_rows(res)
    count = [len(have)]
    prior = set()                                   # lookups made for earlier events of the stream being recorded

    def record(job, row, p):
        arm62.append(res, row)
        for r in p:
            arm62.append(priv, r)
        have[row["id"]] = row
        v = score62b.judge(job, row, frozenset(prior))
        prior.update(score62b.key(c) for c in row["calls"] if c["name"] in score62b.LOOKUPS)
        print(arm, row["id"], row["case"], "right" if v["right"] else "WRONG " + "; ".join(v["why"]), row["end"],
              [c["name"] for c in row["calls"]], row["seconds"], "thermal", row["thermal_before"], flush=True)
        count[0] += 1
        if count[0] == 15 and not fut.exists():
            alljobs = {**jobs, **{e["id"]: e for s in data["streams"] for e in s["events"]}}
            pri = score62b.priors(data, have)
            right = sum(score62b.judge(alljobs[i], r, pri.get(i, frozenset()))["right"] for i, r in have.items())
            fut.write_text(json.dumps({"after": 15, "right": right, "stop": right == 0}) + "\n")

    def stopped():
        return fut.exists() and json.loads(fut.read_text())["stop"]

    def failed(jid, e, attempt):
        """A failed call is never an answer: it is logged, and the decision (or the whole stream) is asked again."""
        arm62.append(out / "errors.jsonl", {"id": jid, "attempt": attempt, "error": f"{type(e).__name__}: {e}"[:600],
                                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if attempt >= ATTEMPTS:
            print(f"STOPPED: the call for {jid} failed {attempt} times ({type(e).__name__}); nothing is scored for it", flush=True)
            sys.exit(4)
        print(f"retry: the call for {jid} failed (attempt {attempt}: {str(e)[:120]}); asking it again", flush=True)
        time.sleep(5)

    for jid in data["order"]:
        if stopped():
            break
        if jid in have:
            continue
        for attempt in range(1, ATTEMPTS + 1):
            try:
                row, p = decide(arm, world, jobs[jid], [system])
                break
            except Exception as e:
                failed(jid, e, attempt)
        prior.clear()                                # a single job is its own conversation
        record(jobs[jid], row, p)
    for s in data["streams"]:
        if stopped():
            break
        ids = [e["id"] for e in s["events"]]
        if all(i in have for i in ids):
            continue
        for attempt in range(1, ATTEMPTS + 1):        # a stream is rerun whole if a call in it failed
            messages, rows = [system], []
            try:
                for e in s["events"]:
                    row, p = decide(arm, world, e, messages)
                    row["stream"] = s["id"]
                    rows.append((e, row, p))
                break
            except Exception as ex:
                failed(s["id"], ex, attempt)
        prior.clear()
        for e, row, p in rows:
            record(e, row, p)
        prior.clear()
    if stopped():
        print(f"arm {arm} stopped by the futility rule: right on none of its first 15 decisions", flush=True)
    if priv.exists():
        (out / "reasoning.sha256").write_text(f"{arm62.sha256_file(priv)}  reasoning-{arm}.jsonl (private)\n")
    print("done", arm, out, flush=True)


def port_busy() -> bool:
    try:
        urllib.request.urlopen(HOST + "/health", timeout=3).read()
        return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError):
        return False


def wait_ready(proc, seconds=300) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc.poll() is not None:
            return False
        try:
            if b'"ok"' in urllib.request.urlopen(HOST + "/health", timeout=3).read():
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_all(run: Path, private: Path, jobs_file: Path, arms: list, unpinned: bool):
    run.mkdir(parents=True, exist_ok=True)
    meta = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv, "arms": arms,
            "python": sys.version, "macos": arm62._run("sw_vers", "-productVersion"),
            "ollama_version": arm62._run("ollama", "--version"), "ollama_ps_before": arm62._run("ollama", "ps"),
            "thermal_start": arm62.thermal(), "max_tokens": MAX_TOKENS, "max_turns": MAX_TURNS, "seed": SEED,
            "jobs_file": jobs_file.name, "jobs_sha256": arm62.sha256_file(jobs_file), "unpinned": unpinned,
            "server_commit": arm62._run("git", "-C", str(SERVER_REPO), "rev-parse", "HEAD"), "arm_rc": {}, "server_args": {}}

    def save():
        (run / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")

    need_ink, need_var = "I" in arms, any(a.startswith("V") for a in arms)
    print("hashing the model files (about a minute) ...", flush=True)
    meta["ink_sha256"] = arm62.sha256_file(INK) if need_ink else None
    meta["var_sha256"] = arm62.sha256_file(VAR) if need_var else None
    meta["pinned"] = {"jobs_file": unpinned or meta["jobs_sha256"] == TEST_SHA256,
                      "server_commit": meta["server_commit"] == SERVER_COMMIT,
                      "inkling_file": not need_ink or meta["ink_sha256"] == INK_SHA256,
                      "research_file": not need_var or meta["var_sha256"] == VAR_SHA256}
    meta["port_was_free"] = not port_busy()
    save()
    if not all(meta["pinned"].values()) or not meta["port_was_free"]:
        print("STOP: a pinned input does not match, or something already answers on the side port:", meta["pinned"],
              "port free:", meta["port_was_free"], flush=True)
        sys.exit(2)
    for arm in arms:
        env = {**os.environ, "PYTHONPATH": str(arm62.HOOK), "AUDIT54_LOG": str(run / f"audit-{arm}.jsonl")}
        argv = [sys.executable, __file__, "--arm", arm, "--run", str(run), "--private", str(private), "--jobs", str(jobs_file)]
        args = server_args(arm)
        meta["server_args"][arm] = args
        if args is None:
            rc = subprocess.run(argv, env=env).returncode
        else:
            log = open(run / f"server-{arm}.log", "ab")
            server = subprocess.Popen([str(SERVER), *args], stdout=log, stderr=subprocess.STDOUT)
            try:
                if not wait_ready(server):
                    print("STOP: the server did not become ready; see", run / f"server-{arm}.log", flush=True)
                    sys.exit(3)
                rc = subprocess.run(argv, env=env).returncode
            finally:
                server.terminate()
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    server.kill()
                log.close()
                time.sleep(3)
        meta["arm_rc"][arm] = rc
        save()
        if rc != 0:
            print(f"STOPPED EARLY in arm {arm}; resume with the same --run", flush=True)
            sys.exit(rc)
    meta.update({"ollama_ps_after": arm62._run("ollama", "ps"), "thermal_end": arm62.thermal(),
                 "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    save()
    print("all done", run, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="driver: checks, then each arm with its server")
    ap.add_argument("--arm", help="child: one arm; its server must be up")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--run", required=True)
    ap.add_argument("--private", required=True, help="folder outside the repository for the reasoning text")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--unpinned", action="store_true", help="dev and smoke files: the jobs file's hash is recorded, not checked")
    a = ap.parse_args()
    run, private = Path(a.run).resolve(), Path(a.private).expanduser().resolve()
    repo = KIT.parent.parent.parent.parent
    if repo in private.parents or private == repo:
        sys.exit("--private must be outside the repository")
    if a.all:
        run_all(run, private, Path(a.jobs).resolve(), a.arms.split(","), a.unpinned)
    else:
        run_arm(a.arm, Path(a.jobs), run / a.arm, private)


if __name__ == "__main__":
    main()
