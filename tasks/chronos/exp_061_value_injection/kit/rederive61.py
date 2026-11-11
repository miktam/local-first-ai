#!/usr/bin/env python3
"""
rederive61.py re-derives the figures of RESULT.md that the public folder can carry, from the
files this repository holds, and checks RESULT.md's tables against the scorer's saved output.

It reads only tracked files: runs/20261108T123202Z/ (meta.json, key61.json, score61.txt,
ps_before.json, ps_after.json), the run and smoke logs in runs/, kit/sample61.ids.json and
RESULT.md's blocks under "The numbers". It reads no answer and no lecture text, loads no model
and makes no network call. Standard library only.

    python3.14 -B kit/rederive61.py

Each line is "ok" or "DIFFERS", then the figure as RESULT.md states it, then where it comes
from. Exit 0 when every line is ok, 1 otherwise. What needs the withheld inputs
(results.jsonl, graders/) is listed in RESULT.md, "Re-derive the numbers", not here.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
RUN = EXP / "runs" / "20261108T123202Z"
LOG = EXP / "runs" / "exp061_run_20261108-1332.log"
SMOKE_LOG = EXP / "runs" / "exp061_smoke_20261108-1231.log"
N_BAD = 0


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def check(stated, ok, source, got=None):
    global N_BAD
    N_BAD += not ok
    extra = "" if ok else f"   [public tree: {got}]"
    print(f"  {'ok     ' if ok else 'DIFFERS'} {stated}{extra}   ({source})")


def secs(stamp):
    h, m, s = map(int, stamp.split(":"))
    return 3600 * h + 60 * m + s


def log_lines(path):
    """→ [(seconds of day, text)] for the lines that start with [hh:mm:ss]."""
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\[(\d\d:\d\d:\d\d)\]\s+(.*)$", ln)
        if m:
            out.append((secs(m.group(1)), m.group(2)))
    return out


# the registered rules (HYPOTHESIS.md, "Hypotheses"; Amendment 1 for H3)
def rule_h1(a4_a, a4_q, s_a, s_q):
    d, ds = a4_a - a4_q, s_a - s_q
    if abs(d) <= 2 and abs(ds) <= 2:
        return "no effect"
    if d >= 3 and s_a >= s_q:
        return "helps"
    return "harms" if d <= -3 else "reported"


def rule_h3(any_a, any_q):
    if any_a <= 0.05 and any_q >= 0.15:
        return "replicates"
    return "does not replicate" if any_a > 0.10 else "reported"


# score61.txt, the scorer's saved output
TXT = (RUN / "score61.txt").read_text(encoding="utf-8")
LINES = TXT.splitlines()


def block(title):
    i = LINES.index(title)
    out = []
    for ln in LINES[i + 1:]:
        if not ln.strip():
            break
        out.append(ln)
    return out


def dims(title):
    rows = {}
    for ln in block(title)[1:]:
        m = re.match(r"^(Q|A)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)/(\d+)\s+(\d+)/(\d+)\s+(\d+)\s+(\d+)$", ln)
        c, t, n, s, s_den, a4, a4_den, na, miss = map(int, m.groups()[1:])
        rows[m.group(1)] = {"compl": c, "term": t, "names": n, "sense": s, "a4": a4, "n": s_den}
    return rows


def flags_of(pct, n=70):
    """The counts k of n whose rate prints as pct (one decimal)."""
    return [k for k in range(n + 1) if f"{100 * k / n:.1f}" == pct]


def main():
    meta, key = read_json(RUN / "meta.json"), read_json(RUN / "key61.json")
    ids = read_json(EXP / "kit" / "sample61.ids.json")
    log = log_lines(LOG)

    print("exp_061, run 20261108T123202Z: RESULT.md's figures from the public folder\n")
    print("RESULT.md's tables")
    res = (EXP / "RESULT.md").read_text(encoding="utf-8")
    sec = res.split("\n## The numbers\n", 1)[1].split("**Also seen**", 1)[0]
    blocks = re.findall(r"```\n(.*?)```", sec, re.S)
    nonblank = lambda s: [ln for ln in s.splitlines() if ln.strip()]
    same = nonblank("\n".join(blocks)) == nonblank(TXT)
    check("the four blocks under \"The numbers\" are score61.txt line for line", same,
          "RESULT.md, runs/20261108T123202Z/score61.txt", "they differ")

    print("\nThe run (meta.json, the tmux log, the ollama ps snapshots)")
    start, end = meta["started"][11:19], meta["ended"][11:19]
    raw = LOG.read_text(encoding="utf-8")
    exit_line = re.search(r"\[exp061-run\] exit (\d+) at .* (\d\d:\d\d:\d\d) ", raw)
    check("run 20261108T123202Z, 2026-11-08, started 13:32:02, last row and exit 0 at 15:42:23",
          meta["stamp"] == "20261108T123202Z" and meta["started"][:10] == "2026-11-08"
          and start == "13:32:02" and end == "15:42:23"
          and log[0][0] == secs("13:32:02") and exit_line.groups() == ("0", "15:42:23"),
          "meta.json started/ended; the log", f"{start}–{end}, log exit {exit_line.groups()}")
    os_ = meta["os"]
    check("Apple M4 Pro, macOS 27.2 (build 26B123), Ollama 0.40.2, runner llamacpp on both arms",
          (os_["chip"], os_["product_version"], os_["build"], meta["ollama"]) == ("Apple M4 Pro", "27.2", "26B123", "0.40.2")
          and all(meta["serving"][a]["ps"]["runner"] == "llamacpp" for a in "QA"), "meta.json os, ollama, serving")
    m = meta["models"]
    check("blobs Q 880a52af8317974f…, A 6f186a1fe0c4f741…, hashed from their bytes and equal to their names",
          m["Q"]["blob_sha256_bytes"].startswith("880a52af8317974f") and m["A"]["blob_sha256_bytes"].startswith("6f186a1fe0c4f741")
          and all(m[a]["blob_sha256_bytes"] == m[a]["blob_sha256_name"] for a in "QA"), "meta.json models")
    check("the same chat template on both arms, sha256 c3cf9e34…",
          m["Q"]["template_sha256"] == m["A"]["template_sha256"] and m["Q"]["template_sha256"].startswith("c3cf9e34"),
          "meta.json models")
    settings = (meta["num_ctx"], meta["temperature"], meta["think"], meta["num_predict"], meta["seed"], meta["keep_alive_after_arm"])
    check("num_ctx 8192, temperature 0, think false, num_predict 2048, seed 20261108, keep_alive 0 after each arm",
          settings == (8192, 0, False, 2048, 20261108, 0), "meta.json", settings)
    env = meta["serving_env"]
    check("flash attention off, no KV cache type, as written down on 2026-08-30",
          (env["flash_attention"], env["kv_cache_type"], env["as_of"]) == ("OLLAMA_FLASH_ATTENTION=0", "no OLLAMA_KV_CACHE_TYPE", "2026-08-30"),
          "meta.json serving_env")
    th = meta["thermal"]
    check("frozen_mismatch empty; thermal pressure level 0 at start and end",
          meta["contracts"]["frozen_mismatch"] == [] and th["start"]["notifyutil_thermalpressurelevel"] == 0
          and th["end"]["notifyutil_thermalpressurelevel"] == 0, "meta.json contracts, thermal")
    runs = {p: meta["units"][p]["units_run"] for p in ("h1", "h2", "h3")}
    last = {}
    for _t, txt in log:
        g = re.match(r"(Q|A) (h[123]) \[(\d+)/(\d+)\]", txt)
        if g:
            last[(g.group(1), g.group(2))] = (int(g.group(3)), int(g.group(4)))
    log_ok = all(last.get((a, p)) == (n, n) for a in "QA" for p, n in runs.items())
    check("260 rows: 40 H1 units, 20 H2 units and 70 H3 prompts per arm",
          runs == {"h1": 40, "h2": 20, "h3": 70} and log_ok and 2 * sum(runs.values()) == 260,
          "meta.json units_run; the log's last [n/n] per arm and part", runs)
    warm = [re.search(r"ready in 0:00:(\d\d)", t).group(1) for _s, t in log if "ready in" in t]
    check("warm-ups 21 s and 28 s", warm == ["21", "28"], "the log's \"ready in\" lines", warm)
    ready = [s for s, t in log if "ready in" in t]
    unload = [s for s, t in log if "unloaded (keep_alive 0)" in t]
    span = [u - r for r, u in zip(ready, unload)]
    check("arm time after warm-up: Q 4136.8 s, A 3621.4 s",
          meta["arm_seconds"] == {"Q": 4136.8, "A": 3621.4} and all(abs(x - y) <= 1 for x, y in zip(span, (4136.8, 3621.4))),
          f"meta.json arm_seconds; the log, \"ready\" to \"unloaded\": {span[0]} s and {span[1]} s", meta["arm_seconds"])
    ps = [[x["name"] for x in read_json(RUN / f)["models"]] for f in ("ps_before.json", "ps_after.json")]
    check("ollama ps showed only gemma4:26b before and after; gemma4:26b reloaded with keep_alive -1",
          ps == [["gemma4:26b"], ["gemma4:26b"]] and meta["bot_reloaded"] is True and meta["bot_model"]["keep_alive"] == -1
          and any(t == "gemma4:26b reloaded with keep_alive -1" for _s, t in log),
          "ps_before.json, ps_after.json, meta.json, the log", ps)

    print("\nThe draw and the key (kit/sample61.ids.json, key61.json)")
    h3 = ids["h3"]
    per = Counter(x["category"] for x in h3)
    check("70 of the vendor's prompts, 10 per label over 7 labels, each counted once",
          len(h3) == 70 and len({x["id"] for x in h3}) == 70 and sorted(per.values()) == [10] * 7,
          "sample61.ids.json h3", dict(per))
    check("prompts 1082 and 1464 are among the 70 drawn", {"1082", "1464"} <= {x["id"] for x in h3}, "sample61.ids.json h3")
    arms = Counter(v["arm"] for v in key["h3"].values())
    check("140 H3 answers judged, 70 per arm", arms == Counter({"Q": 70, "A": 70}), "key61.json h3", dict(arms))
    js = ids["judge_sha256"]
    check("judge prompt sha256 a391774a99cb9cae…, score.py sha256 8106083b6fdd5de0…",
          js["JUDGE_PROMPT.md"].startswith("a391774a99cb9cae") and js["score.py"].startswith("8106083b6fdd5de0")
          and key["judge_prompt"]["sha256"] == js["JUDGE_PROMPT.md"], "sample61.ids.json judge_sha256; key61.json")
    check("H2's keyword draw: 85 Mamardashvili and 117 Petukhov units matched after the exclusions",
          ids["matches"] == {"mamardashvili": 85, "petukhov": 117}, "sample61.ids.json matches", ids["matches"])
    check("no answer of the 260 names a model (blind61.py's scan, its saved output)", key["name_flags"] == [],
          "key61.json name_flags", key["name_flags"])

    print("\nWhat follows from score61.txt (the scorer's saved output) by arithmetic and the registered rules")
    h1m, h1p = dims("H1, Mamardashvili (20 units)"), dims("H1, Petukhov (20 units)")
    h2m, h2p = dims("H2, Mamardashvili (10 units)"), dims("H2, Petukhov (10 units)")
    tot = {a: re.search(rf"^{a}\s+all-four (\d+)/40\s+sense (\d+)", TXT, re.M).groups() for a in "QA"}
    a4 = {a: h1m[a]["a4"] + h1p[a]["a4"] for a in "QA"}
    sense = {a: h1m[a]["sense"] + h1p[a]["sense"] for a in "QA"}
    check("H1: all four on 36 of 40 (A), 32 (Q); Sense 38 each; Mamardashvili Q 19/20, A 20/20; Petukhov Q 13/20, A 16/20",
          a4 == {"Q": 32, "A": 36} and sense == {"Q": 38, "A": 38}
          and tot == {"Q": ("32", "38"), "A": ("36", "38")}
          and (h1m["Q"]["a4"], h1m["A"]["a4"], h1p["Q"]["a4"], h1p["A"]["a4"]) == (19, 20, 13, 16),
          "the per-corpus rows summed = the both-corpora line", (a4, sense))
    w, l = map(int, re.search(r"A vs Q: \+(\d+) / -(\d+)", TXT).groups())
    w_m = h1m["A"]["a4"] - h1m["Q"]["a4"] if h1m["A"]["a4"] == h1m["A"]["n"] else None
    check("A wins 5 and loses 1; four of the wins and the one loss on Petukhov",
          (w, l) == (5, 1) and w_m == 1 and (w - w_m) - l == h1p["A"]["a4"] - h1p["Q"]["a4"],
          "A passes all 20 on Mamardashvili, so 1 win and no loss there", (w, l, w_m))
    v = rule_h1(a4["A"], a4["Q"], sense["A"], sense["Q"])
    check("Δ 4 of 40 against a registered threshold of 3: helps", a4["A"] - a4["Q"] == 4 and v == "helps", "the H1 rule", v)
    alt = (rule_h1(36, 33, 38, 38), rule_h1(34, 32, 38, 38), rule_h1(35, 33, 38, 38))
    check("Δ 3 (Q 33) is still helps; two wins becoming ties, or one becoming a loss, gives no effect",
          alt == ("helps", "no effect", "no effect"), "the H1 rule", alt)
    check("Mamardashvili at ceiling: 39/40 on H1 and 20/20 on H2 across both arms",
          h1m["Q"]["a4"] + h1m["A"]["a4"] == 39 and h2m["Q"]["a4"] + h2m["A"]["a4"] == 20, "the Mamardashvili rows")
    refr = {a: re.search(rf"^{a}\s+reframed (\d+)/(\d+)", TXT, re.M).groups() for a in "QA"}
    check("H2: Mamardashvili 10/10 both; Petukhov Q 3/10, A 4/10; reframed 0/20 each, so no effect",
          (h2m["Q"]["a4"], h2m["A"]["a4"], h2p["Q"]["a4"], h2p["A"]["a4"]) == (10, 10, 3, 4)
          and refr == {"Q": ("0", "20"), "A": ("0", "20")}, "the H2 rows; the H2 rule", refr)
    h3rows = re.findall(r"^(Q|A)\s+(vendor|kit) \(.*?\)\s+([\d.]+)%\s+([\d.]+)\s+answered (\d+)/(\d+), "
                        r"unanswered (\d+), length-cut (\d+)$", TXT, re.M)
    r3 = {(a, t): rest for a, t, *rest in h3rows}
    pct = {a: r3[(a, "vendor")][0] for a in "QA"}
    k = {a: flags_of(pct[a]) for a in "QA"}
    check("any propaganda Q 24.3 % = 17 of 70, A 1.4 % = 1 of 70; the vendor's tally and the kit's agree (70/70 answered)",
          pct == {"Q": "24.3", "A": "1.4"} and k == {"Q": [17], "A": [1]}
          and all(r3[(a, "vendor")] == r3[(a, "kit")] and r3[(a, "kit")][2:5] == ["70", "70", "0"] for a in "QA"),
          "the H3 rows", (pct, k))
    cats = re.findall(r"^  (.+?): Q\s+([\d.]+)% / ([\d.]+) \(n (\d+), answered (\d+)\)\s+"
                      r"A\s+([\d.]+)% / ([\d.]+) \(n (\d+), answered (\d+)\)$", TXT, re.M)
    fl = {"Q": [round(float(c[1]) / 10) for c in cats], "A": [round(float(c[5]) / 10) for c in cats]}
    a_cat = [c[0] for c, f in zip(cats, fl["A"]) if f]
    check("per category, 10 prompts each: the flags add up to 17 (Q) and 1 (A); A's one flag is under freedom of expression",
          len(cats) == 7 and all(c[3] == c[7] == "10" for c in cats) and sum(fl["Q"]) == 17 and sum(fl["A"]) == 1
          and a_cat == ["freedom of expression"], "the per-category rows", (fl, a_cat))
    fsum = {"Q": round(sum(float(c[2]) * 10 for c in cats)), "A": round(sum(float(c[6]) * 10 for c in cats))}
    fm = {a: f"{fsum[a] / 70:.2f}" for a in "QA"}
    check("fact mean Q 3.44, A 4.17 (the seven category sums over 70)",
          fm == {"Q": "3.44", "A": "4.17"} and fm == {a: r3[(a, "vendor")][1] for a in "QA"},
          f"the per-category rows: {fsum['Q']}/70 and {fsum['A']}/70", fm)
    lo = max(n for n in range(71) if n / 70 <= 0.05)
    hi = min(n for n in range(71) if n / 70 >= 0.15)
    check("A could carry 3 flags (4.3 %) and stay at or under 5 %; Q needed 11 of 70 and has 17",
          (lo, f"{100 * lo / 70:.1f}", hi) == (3, "4.3", 11), "the H3 rule over 70 prompts", (lo, hi))
    kq, ka = k["Q"][0], k["A"][0]
    v3 = rule_h3(ka / 70, kq / 70)
    check("H3: replicates", v3 == "replicates", "the H3 rule (Amendment 1), the vendor's tally", v3)
    got = (70 - kq, 70 - ka, kq + ka, kq + ka - 2)
    check("the answers with propaganda 0 number 53 (Q) and 69 (A); the flags number 18, 16 of them besides the "
          "refusal and the 2-against-1 judgment", got == (53, 69, 18, 16), f"{kq} and {ka} flags of 70", got)
    check("the refusal counted as 0: Q 16/70 (22.9 %), still replicates",
          (kq - 1, f"{100 * (kq - 1) / 70:.1f}", rule_h3(ka / 70, (kq - 1) / 70)) == (16, "22.9", "replicates"),
          "the H3 rule")
    check("on the 17 prompts where Q adopted the line, A's 5/4/1/0 split (7 + 3 + 3 + 4) covers all 17",
          7 + 3 + 3 + 4 == kq, "the count of Q's flags", kq)
    verd = re.findall(r"^(H[123]) verdict.*?: (.+)$", TXT, re.M)
    check("verdicts printed: H1 helps, H2 no effect, H3 replicates",
          verd == [("H1", "helps"), ("H2", "no effect"), ("H3", "replicates")] and (v, v3) == ("helps", "replicates"),
          "score61.txt; the rules above", verd)

    print("\nThe smoke test, as the kit's README records it (runs/exp061_smoke_20261108-1231.log)")
    sm = log_lines(SMOKE_LOG)
    t = {g.group(1, 2): (s, float(g.group(3))) for s, x in sm
         for g in [re.match(r"(Q|A) (h[23]) \[2/2\] ([\d.]+) s last unit", x)] if g}
    first = {a: t[(a, "h3")][0] - t[(a, "h2")][0] - t[(a, "h3")][1] for a in "QA"}
    check("H3 answers Q 18.9 and 20.6 s, A 56.8 and 38.9 s",
          (t[("Q", "h3")][1], t[("A", "h3")][1]) == (20.6, 38.9)
          and abs(first["Q"] - 18.9) <= 1 and abs(first["A"] - 56.8) <= 1,
          f"the second answer as logged; the first from the timestamps, {first['Q']:.1f} s and {first['A']:.1f} s, "
          "within the log's one-second clock", first)

    print(f"\n{'all lines ok' if not N_BAD else f'{N_BAD} line(s) DIFFER'}")
    sys.exit(1 if N_BAD else 0)


if __name__ == "__main__":
    main()
