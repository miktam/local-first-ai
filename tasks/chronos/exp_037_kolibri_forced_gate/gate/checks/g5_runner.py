# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06; build task W5c).
"""The exp_037 G5 family: G5-D32, G5-R1, G5-BP-lean and phase 6's R1 anchor
(DESIGN §3.12-§3.14, §2.9; G1 item 8). Measurement only: every decision is
gate/rules.py's (g5_d32, g5_r1, g5_bp, allowed_B), on the dicts made here.

g5_d32(model, module, texts, ranges)                                   §3.12
  The K8 fp32 port (the caller converts it: set_dtype(float32) and
  g4_forced.fp32_on_dequantised_weights). Per range (T1 and T3 positions
  520-1,100, T9 15,000-15,300; hi inclusive), through gate.harness.ForcedTrunk:
    (i)   free, chunk 2,048, recording ids;
    (ii)  forced onto (i)'s ids sorted ascending, chunk 2,048 (the baseline);
    (iii) forced decode: the prefix in 2,048-token chunks, then one token at a time;
    (iv)  forced, chunk 64.
  (iii) and (iv) are forced onto the same sorted ids at every position from 0.
  Csort = KL((i)||(ii)); the comparisons are KL((ii)||(iii)) and KL((ii)||(iv)),
  with (ii)'s lead at every position where the top-1 differs (a rules.tail).
  Mutant 20 runs with comparisons=("decode",).

g5_r1(model, prompts, floor, out_dir, arm=...)                          §3.13
  The production runner at B = 1: runner.generate.make_batch_generator(model,
  1, eos, 256, greedy) through run_cell's admission loop
  (g5_generation.run_admission), each of the 8 G5 prompts alone. "single" is
  prompt + runner tokens teacher-forced alone through gate.harness.logits_at
  (chunk 2,048). Writes <seq>.runner.npy and <seq>.single.npy (fp32
  log-probs at the generated positions) and reduces R-parity, which needs no
  R1: mean KL(single||runner) and the top-1 disagreement, pooled. Mutant 26
  is the same call on the mutant model (mutant=26; its block has needs_r1 False).

g5_bp(model, prompts26, B, out_dir, arm=...)                            §3.14
  make_batch_generator(model, B, ...) with the registered prefill_batch_size
  = min(B, 8), greedy, through the same admission loop; the 26 prompts of
  prompts26(ts) in queue order. Labels: first_wave = the first B inserted,
  mid_run = inserted after the first finish. Writes <seq>.batched.npy and
  <seq>.single.npy; (d) needs R1 and is reduced in phase 6. Mutant 27, the
  required control (decision (b)), is the same call on the mutant model at K8,
  B = 8 (mutant=27); so is probe 22 in tiny mode (mutant=22).

prompts26(ts)            §3.14's layout from thresholds G5BP.layout (real), or
                         the tiny profile's lengths with G5BP.tiny_layout (§5.1)
greedy_block(conts, ts.g5)  the G5 greedy continuations (generate_step) as a block
anchor_sequences(phase_jsons)  every sequence phase 6 must score, both arms
reduce_with_r1(phase_jsons, r1)  phase 6's measured dicts, in rules.py's shapes

Data flow (DESIGN §2.9, §3.13). Each generating call returns a "block"
(schema BLOCK_SCHEMA) that W7 stores in its phase JSON (anywhere under it;
anchor_sequences finds every block by its schema; block ids "<arm>/<check>"
are unique). A block's "needs_r1" says whether phase 6 scores it with R1
(every block but mutant 26's). A block carries the
prompt ids, the generated tokens, the labels and, per log-prob file, its path
relative to work37_dir() ($EXP036_WORK/exp037; no home path enters a record)
and its sha256. Files live under work37_dir()/run/<UTC>/g5/<arm>/<check>/,
<check> in g5_r1, g5_r1_m26, g5_bp_B8, g5_bp_B16, g5_bp_m27 (and g5_bp_m22 for
the tiny-mode probe) (check_dir_name).
Files are created, never overwritten, and every load checks the sha256 first.
Phase 6: anchor = anchor_sequences(phase_jsons); r1 =
ref_drivers.r1_rows_pass(bf16_dir, anchor["seqs"], anchor["rows"]); then
reduce_with_r1(phase_jsons, r1, anchor). Identical sequences (prompt +
tokens) are scored once.

Every KL is common.kl_rows (a float64 log-softmax of the fp32 inputs, DESIGN
§3.0). The log-prob files hold fp32 log-probs; kl_rows re-normalises them in
float64, so no KL is taken from unnormalised fp32 log-probabilities.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from gate import common
from gate.checks import g5_generation as g5
from gate.rules import tail

BLOCK_SCHEMA = "exp037 g5 sequences v1"
ANCHOR_SCHEMA = "exp037 g5 anchor v1"
REDUCE_SCHEMA = "exp037 g5 reductions v1"
D32_SCHEMA = "exp037 g5 d32 v1"
CHECKS = ("greedy", "g5_r1", "g5_bp")
GEN_KIND = {"g5_r1": "runner", "g5_bp": "batched"}  # the generated side's file kind
SINGLE = "single"
SUBSETS = ("first_wave", "mid_run")
LOGPROB_ROOT = "$EXP036_WORK/exp037"
SAMPLER = "greedy: argmax of the fp32 log-probs (not the production make_vllm_sampler; DESIGN §3.17)"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class G5DataError(RuntimeError):
    """The G5 data do not hang together (a missing R1 row, a file whose sha256
    differs, a duplicated block): the gate cannot vouch for a value (exit 3)."""


def _th(th: dict | None) -> dict:
    return th if th is not None else common.load_thresholds()


# ---------------------------------------------------------------------------
# The 26 prompts of G5-BP-lean (§3.14; tiny §5.1)
# ---------------------------------------------------------------------------


def prompts26(ts, th: dict | None = None) -> list:
    """The 26 prompts in queue order A_0..A_11, L1, L2, B_0..B_11.

    A_j = ids[order[j % 8]][:L_j] and B_j = ids[order[(j + 4) % 8]][:L_j],
    each with max_tokens M_j; L1 and L2 are T9 slices that cross a prefill
    chunk. Real (ts.profile.name != "tiny"): L_j, M_j, L1 and L2 from
    thresholds G5BP.layout, whose totals (16,916 prompt tokens, 792 generated
    positions) are checked. Tiny: L_j and M_j from the tiny profile
    (batch_lengths, batch_max_tokens), L1 and L2 from G5BP.tiny_layout.
    Entries: {"seq", "group", "j", "text", "lo", "hi", "ids", "max_tokens"}."""
    t = _th(th)["G5BP"]
    lay = t["layout"]
    tiny = ts.profile.name == "tiny"
    if tiny:
        lengths, mtok = list(ts.profile.batch_lengths), list(ts.profile.batch_max_tokens)
        l1, l2 = t["tiny_layout"]["L1"], t["tiny_layout"]["L2"]
    else:
        lengths, mtok = list(lay["lengths"]), list(lay["max_tokens"])
        l1, l2 = lay["L1"], lay["L2"]
    order, off = list(lay["text_order"]), int(lay["b_text_offset"])
    if len(lengths) != 12 or len(mtok) != 12:
        raise ValueError(f"G5-BP-lean needs 12 lengths and 12 max_tokens, got {len(lengths)} and {len(mtok)}")

    def entry(seq, group, j, text, ids, lo, hi, m):
        sl = list(ids[lo:hi])
        if len(sl) != hi - lo or hi <= lo:
            raise ValueError(f"{seq}: {text}[{lo}:{hi}] has {len(sl)} tokens (the text is {len(ids)} long)")
        if int(m) < 1:
            raise ValueError(f"{seq}: max_tokens {m}")
        return {"seq": seq, "group": group, "j": j, "text": text, "lo": int(lo), "hi": int(hi),
                "ids": [int(x) for x in sl], "max_tokens": int(m)}

    out = [entry(f"A{j:02d}", "A", j, order[j % 8], ts.ids[order[j % 8]], 0, lengths[j], mtok[j]) for j in range(12)]
    for name, l in (("L1", l1), ("L2", l2)):
        a, b = l["t9_ids"]
        out.append(entry(name, "L", None, "T9", ts.t9, a, b, l["max_tokens"]))
    out += [entry(f"B{j:02d}", "B", j, order[(j + off) % 8], ts.ids[order[(j + off) % 8]], 0, lengths[j], mtok[j])
            for j in range(12)]
    if not tiny:
        n_prompt, n_gen = sum(len(e["ids"]) for e in out), sum(e["max_tokens"] for e in out)
        if (n_prompt, n_gen) != (lay["prompt_tokens"], lay["generated_positions"]):
            raise ValueError(f"the 26-prompt layout has {n_prompt} prompt tokens and {n_gen} generated positions; "
                             f"thresholds say {lay['prompt_tokens']} and {lay['generated_positions']}")
    return out


# ---------------------------------------------------------------------------
# Log-prob files (§2.9, §3.13)
# ---------------------------------------------------------------------------


def check_dir_name(check: str, *, B: int | None = None, mutant: int | None = None) -> str:
    """g5_r1, g5_r1_m26, g5_bp_B8, g5_bp_B16, g5_bp_m27 (g5_bp_m22: the tiny-mode probe)."""
    if check == "g5_r1":
        return "g5_r1" if mutant is None else f"g5_r1_m{int(mutant)}"
    if check == "g5_bp":
        return f"g5_bp_m{int(mutant)}" if mutant is not None else f"g5_bp_B{int(B)}"
    raise ValueError(f"no log-prob directory for check {check!r}")


def g5_out_dir(run_utc: str, arm: str, check_dir: str) -> Path:
    """work37_dir()/run/<UTC>/g5/<arm>/<check_dir>/ (DESIGN §2.9)."""
    for part in (run_utc, arm, check_dir):
        if not _NAME.match(str(part)):
            raise ValueError(f"not a plain path component: {part!r}")
    return common.work37_dir() / "run" / str(run_utc) / "g5" / str(arm) / str(check_dir)


def logprob_root() -> Path:
    return common.work37_dir().expanduser().resolve()


def _prepare_out_dir(out_dir) -> Path:
    from gate.checks.ref_drivers import require_under_work37

    out = require_under_work37(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_logprobs(out_dir, seq: str, kind: str, arr: np.ndarray) -> dict:
    """Write <out_dir>/<seq>.<kind>.npy (fp32 [n, V]) as a new file and return
    its entry {"path" (relative to work37_dir()), "sha256", "shape", "dtype",
    "bytes"}. Refuses an out_dir outside work37_dir() and an existing file."""
    if not _NAME.match(str(seq)) or kind not in ("runner", "batched", SINGLE):
        raise ValueError(f"bad log-prob file name {seq!r}.{kind!r}")
    out = _prepare_out_dir(out_dir)
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"log-probs must be [n, V]; got shape {a.shape}")
    path = out / f"{seq}.{kind}.npy"
    with open(path, "xb") as f:  # never overwrite (BUILD_SPEC §2 append-only)
        np.save(f, a)
    rel = path.relative_to(logprob_root()).as_posix()
    return {"path": rel, "sha256": common.sha256_file(path), "shape": list(a.shape), "dtype": "float32",
            "bytes": path.stat().st_size}


def logprob_path(entry: dict, root: Path | None = None) -> Path:
    rel = Path(entry["path"])
    if rel.is_absolute() or ".." in rel.parts:
        raise G5DataError(f"log-prob path {entry['path']!r} is not relative to {LOGPROB_ROOT}")
    return Path(root or logprob_root()) / rel


def load_logprobs(entry: dict, root: Path | None = None) -> np.ndarray:
    """The array of a log-prob entry, after its sha256 and shape are checked."""
    p = logprob_path(entry, root)
    if not p.is_file():
        raise G5DataError(f"log-prob file {entry['path']} is missing")
    sha = common.sha256_file(p)
    if sha != entry["sha256"]:
        raise G5DataError(f"log-prob file {entry['path']}: sha256 {sha[:16]} differs from the record's "
                          f"{entry['sha256'][:16]}")
    a = np.load(p)
    if list(a.shape) != list(entry["shape"]) or a.dtype != np.float32:
        raise G5DataError(f"log-prob file {entry['path']}: {a.dtype} {list(a.shape)}, the record says "
                          f"{entry['dtype']} {entry['shape']}")
    return a


def find_blocks(obj) -> list:
    """Every G5 block (schema BLOCK_SCHEMA) anywhere in `obj` (a phase JSON, a
    list of them, or a dict of them); block ids must be unique."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("schema") == BLOCK_SCHEMA:
                found.append(o)
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                if isinstance(v, (dict, list, tuple)):
                    walk(v)

    walk(obj)
    ids = [b["id"] for b in found]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise G5DataError(f"G5 blocks appear more than once: {dup}")
    return found


def logprob_files(phase_jsons) -> list:
    """[{"block", "seq", "kind", "path", "sha256"}] for every log-prob file the
    phase JSONs name: the gate record's g5_logprob_files (DESIGN §3.16)."""
    out = []
    for b in find_blocks(phase_jsons):
        for s in b["sequences"]:
            for kind, e in sorted((s.get("files") or {}).items()):
                out.append({"block": b["id"], "seq": s["seq"], "kind": kind, "path": e["path"], "sha256": e["sha256"]})
    return out


def verify_logprob_files(phase_jsons, root: Path | None = None) -> dict:
    """Every log-prob file the phase JSONs name exists with its sha256:
    {"n_files", "problems": [...]} (G1 item 8)."""
    problems, files = [], logprob_files(phase_jsons)
    for f in files:
        p = logprob_path(f, root)
        if not p.is_file():
            problems.append(f"{f['path']}: missing")
        elif common.sha256_file(p) != f["sha256"]:
            problems.append(f"{f['path']}: sha256 differs")
    return {"n_files": len(files), "problems": problems}


# ---------------------------------------------------------------------------
# Generation through the runner's construction, and the single path
# ---------------------------------------------------------------------------


def _logsoftmax32(rows: np.ndarray, chunk: int = 256) -> np.ndarray:
    out = np.empty(rows.shape, dtype=np.float32)
    for a in range(0, rows.shape[0], chunk):
        out[a:a + chunk] = common.log_softmax64(rows[a:a + chunk]).astype(np.float32)
    return out


def single_logprobs(model, prompt_ids, tokens, chunk: int = 2048) -> np.ndarray:
    """fp32 log-probs [n, V] of `tokens` after `prompt_ids`, teacher-forced
    alone through gate.harness.logits_at (the port's own single-sequence
    cache, chunks of `chunk`): row j is the distribution generated token j was
    drawn from (position P - 1 + j)."""
    from gate.harness import logits_at

    P, n = len(prompt_ids), len(tokens)
    if n == 0:
        raise ValueError("no generated tokens")
    seq = list(prompt_ids) + list(tokens[:-1])
    return _logsoftmax32(logits_at(model, seq, np.arange(P - 1, P - 1 + n), chunk))


def _pair_stats(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(KL(p||q) per position by common.kl_rows, top-1 disagreement per position)."""
    return common.kl_rows(p, q), p.argmax(axis=-1) != q.argmax(axis=-1)


def _pooled(kls: list, dis: list) -> dict:
    if not kls:
        return {"n": 0, "mean_kl": float("nan"), "max_kl": float("nan"), "top1_dis": float("nan")}
    k, d = np.concatenate(kls), np.concatenate(dis)
    return {"n": int(k.size), "mean_kl": float(k.mean()), "max_kl": float(k.max()), "top1_dis": float(d.mean())}


def _generate_block(model, items: list, B: int, out_dir, *, check: str, arm: str, mutant, eos, prefill_step_size: int,
                    chunk: int, extra: dict, log) -> dict:
    """Generate `items` through the runner's construction at B, write the
    generated-side and single log-prob files, and return the block."""
    kind = GEN_KIND[check]
    out = _prepare_out_dir(out_dir)
    names = [it["seq"] for it in items]
    if len(set(names)) != len(names) or not all(_NAME.match(n) for n in names):
        raise ValueError(f"sequence names must be unique plain names: {names}")
    prompts = [list(it["ids"]) for it in items]
    max_tokens = [int(it["max_tokens"]) for it in items]
    gen, construction = g5._batch_generator(model, B, eos, max(max_tokens), prefill_step_size)
    generator = {k: getattr(gen, k, None) for k in ("completion_batch_size", "prefill_batch_size", "prefill_step_size",
                                                    "max_kv_size")}

    def on_finish(j, tokens, lps):
        if lps is None or lps.shape[0] != len(tokens):
            raise G5DataError(f"{names[j]}: {0 if lps is None else lps.shape[0]} log-prob rows for {len(tokens)} tokens")
        return write_logprobs(out, names[j], kind, lps)

    log(f"[gate] {check} {arm} B={B}{'' if mutant is None else f' mutant {mutant}'}: {len(items)} sequences")
    res = g5.run_admission(gen, prompts, max_tokens, B, on_finish=on_finish)
    root = logprob_root()
    seqs, kls, dis = [], [], []
    sub = {k: ([], []) for k in SUBSETS + ("other",)}
    for j, it in enumerate(items):
        s = res["seqs"][j]
        toks = s["tokens"]
        single = single_logprobs(model, it["ids"], toks, chunk)
        e_single = write_logprobs(out, names[j], SINGLE, single)
        generated = load_logprobs(s["on_finish"], root)  # the round trip: sha256 and shape checked
        kl, d = _pair_stats(single, generated)
        kls.append(kl), dis.append(d)
        label = "first_wave" if s["first_wave"] else "mid_run" if s["mid_run"] else "other"
        sub[label][0].append(kl), sub[label][1].append(d)
        meta = {k: it[k] for k in it if k not in ("ids", "max_tokens", "seq")}
        seqs.append({"seq": names[j], **meta, "prompt_ids": prompts[j], "prompt_ids_sha256": common.ids_sha256(prompts[j]),
                     "n_prompt": len(prompts[j]), "tokens": toks, "n": len(toks), "max_tokens": max_tokens[j],
                     "finish_reason": s["finish_reason"], "insert_order": s["insert_order"],
                     "insert_step": s["insert_step"], "finish_step": s["finish_step"], "label": label,
                     "files": {kind: s["on_finish"], SINGLE: e_single},
                     f"single_vs_{kind}": {"mean_kl": float(kl.mean()), "max_kl": float(kl.max()),
                                           "top1_dis": float(d.mean())}})
    trace = {"max_live": res["max_live"], "admitted_mid_run": res["admitted_mid_run"], "steps": res["steps"],
             "first_finish_step": res["first_finish_step"],
             **{lab: [s["seq"] for s in seqs if s["label"] == lab] for lab in SUBSETS + ("other",)}}
    block = {"schema": BLOCK_SCHEMA, "id": f"{arm}/{out.name}", "check": check, "arm": arm,
             "mutant": None if mutant is None else int(mutant), "needs_r1": not (check == "g5_r1" and mutant is not None),
             "B": int(B), "construction": construction, "generator": generator, "sampler": SAMPLER,
             "eos": [int(e) for e in eos], "single": f"gate.harness.logits_at, chunk {chunk}",
             "logprob_root": LOGPROB_ROOT, "dir": out.relative_to(root).as_posix(), "sequences": seqs, "trace": trace,
             f"single_vs_{kind}": {"all": _pooled(kls, dis), **{k: _pooled(*v) for k, v in sub.items()}}}
    block.update(extra)
    return block


def g5_r1(model, prompts: list, floor: dict | None, out_dir, *, arm: str, mutant: int | None = None,
          eos=common.EOS_IDS, max_tokens: int | None = None, prefill_step_size: int | None = None,
          chunk: int | None = None, th: dict | None = None, log=print) -> dict:
    """G5-R1 (§3.13) for one arm (or mutant 26 at K8): the runner at B = 1 on
    the G5 prompts ({"id", "ids", ...}; ts.g5), its single path, the log-prob
    files, and R-parity = {"mean_kl": mean KL(single||runner), "top1_dis",
    "n"}, pooled over every generated position. `floor` (phase 4's noise
    floor) is recorded for the record; rules.g5_r1 applies the bound."""
    t = _th(th)["G5R1"]
    B = int(t["B"])
    m = int(t["max_tokens"] if max_tokens is None else max_tokens)
    step = int(t["prefill_step_size"] if prefill_step_size is None else prefill_step_size)
    items = [{"seq": p["id"], "lang": p.get("lang"), "effort": p.get("effort"), "ids": p["ids"], "max_tokens": m}
             for p in prompts]
    block = _generate_block(model, items, B, out_dir, check="g5_r1", arm=arm, mutant=mutant, eos=eos,
                            prefill_step_size=step, chunk=int(chunk or step), extra={"floor": floor}, log=log)
    p = block["single_vs_runner"]["all"]
    block["parity"] = {"mean_kl": p["mean_kl"], "top1_dis": p["top1_dis"], "n": p["n"]}
    return block


def g5_bp(model, prompts26_: list, B: int, out_dir, *, arm: str, mutant: int | None = None, eos=common.EOS_IDS,
          prefill_step_size: int | None = None, chunk: int | None = None, log=print) -> dict:
    """G5-BP-lean (§3.14) at (arm, B), or mutant 27 (or probe 22) at K8, B = 8: the 26
    prompts through the runner's construction (prefill_batch_size = min(B, 8))
    and admission loop, labelled first_wave / mid_run, with their single path
    and log-prob files. The descriptive single-vs-batched KL per subset is
    reduced here; (d) needs R1 and is reduced by reduce_with_r1."""
    from runner import generate

    step = int(generate.PREFILL_STEP_SIZE if prefill_step_size is None else prefill_step_size)
    items = [{k: e[k] for k in ("seq", "group", "j", "text", "lo", "hi", "ids", "max_tokens")} for e in prompts26_]
    return _generate_block(model, items, int(B), out_dir, check="g5_bp", arm=arm, mutant=mutant, eos=eos,
                           prefill_step_size=step, chunk=int(chunk or step), extra={}, log=log)


def greedy_block(conts: list, prompts: list, *, arm: str = "K8") -> dict:
    """The G5 greedy continuations (g5_generation.greedy_continuations, through
    generate_step) as a block, so that phase 6 scores them with the rest."""
    by_id = {p["id"]: p for p in prompts}
    seqs = []
    for c in conts:
        ids = list(by_id[c["id"]]["ids"])
        if common.ids_sha256(ids) != c["prompt_ids_sha256"] or len(ids) != c["prompt_tokens"]:
            raise G5DataError(f"greedy continuation {c['id']}: its prompt does not match the G5 prompt")
        seqs.append({"seq": c["id"], "lang": c.get("lang"), "effort": c.get("effort"), "prompt_ids": ids,
                     "prompt_ids_sha256": c["prompt_ids_sha256"], "n_prompt": len(ids), "tokens": list(c["continuation"]),
                     "n": len(c["continuation"]), "finish_reason": c["finished"], "label": None, "files": {}})
    return {"schema": BLOCK_SCHEMA, "id": f"{arm}/greedy", "check": "greedy", "arm": arm, "mutant": None,
            "needs_r1": True, "B": None, "construction": "mlx_lm.generate.generate_step (registered G5 greedy)",
            "sequences": seqs}


# ---------------------------------------------------------------------------
# Phase 6: the R1 anchor and the reductions
# ---------------------------------------------------------------------------


def anchor_sequences(phase_jsons) -> dict:
    """Every sequence phase 6 must score with R1, from every block that needs
    it, both arms (greedy, G5-R1 K8 and K4, G5-BP-lean K8 and K4, mutant 27;
    probe 22 in tiny mode):
    {"seqs": [prompt + tokens[:-1]], "rows": [sorted generated positions per
    sequence], "keys": [ids_sha256 per sequence], "uses": [{"block", "seq",
    "anchor", "start", "n"}], "n_sequences", "n_rows", "n_tokens"}. Identical
    sequences are scored once, with the union of their positions. Feed it to
    ref_drivers.r1_rows_pass(bf16_dir, anchor["seqs"], anchor["rows"])."""
    index, seqs, rows, keys, uses = {}, [], [], [], []
    for b in find_blocks(phase_jsons):
        if b["check"] not in CHECKS:
            raise G5DataError(f"block {b['id']}: unknown check {b['check']!r}")
        if not b["needs_r1"]:
            continue
        for s in b["sequences"]:
            toks = list(s["tokens"])
            if not toks:
                continue
            fed = [int(x) for x in s["prompt_ids"]] + [int(x) for x in toks[:-1]]
            key = common.ids_sha256(fed)
            k = index.get(key)
            if k is None:
                k = index[key] = len(seqs)
                seqs.append(fed), rows.append(set()), keys.append(key)
            start = len(s["prompt_ids"]) - 1
            rows[k].update(range(start, start + len(toks)))
            uses.append({"block": b["id"], "seq": s["seq"], "anchor": k, "start": start, "n": len(toks)})
    rows = [np.array(sorted(r), dtype=np.int64) for r in rows]
    return {"schema": ANCHOR_SCHEMA, "seqs": seqs, "rows": rows, "keys": keys, "uses": uses,
            "n_sequences": len(seqs), "n_rows": int(sum(r.size for r in rows)), "n_tokens": int(sum(len(s) for s in seqs))}


class _R1Rows:
    """R1's rows by (anchor sequence, position), from r1_rows_pass's result."""

    def __init__(self, r1: dict, anchor: dict):
        lengths = [len(s) for s in anchor["seqs"]]
        got = [int(x) for x in np.asarray(r1["seq_lengths"]).reshape(-1)]
        if got != lengths:
            raise G5DataError(f"the R1 pass ran on {len(got)} sequences that are not the anchor's {len(lengths)}")
        rec = r1.get("record") or {}
        if rec.get("pack_ids_sha256") and lengths:
            want = common.ids_sha256(np.concatenate([np.asarray(s, dtype=np.int64) for s in anchor["seqs"]]))
            if rec["pack_ids_sha256"] != want:
                raise G5DataError("the R1 pass's token ids are not the anchor's")
        self.logits = r1["logits"]
        self.index = {(int(s), int(p)): j for j, (s, p) in
                      enumerate(zip(np.asarray(r1["seq_index"]).reshape(-1), np.asarray(r1["positions"]).reshape(-1)))}

    def rows(self, use: dict, where: str) -> np.ndarray:
        k, a = use["anchor"], use["start"]
        js = [self.index.get((k, p)) for p in range(a, a + use["n"])]
        if any(j is None for j in js):
            raise G5DataError(f"{where}: phase 6 has no R1 row for {sum(j is None for j in js)} of {use['n']} "
                              "generated positions")
        return np.asarray(self.logits[np.asarray(js, dtype=np.int64)], dtype=np.float32)


def _anchor_d(acc: list) -> dict:
    """{"n", "mean_kl_r1_<gen>", "mean_kl_r1_single"} from [(kl_gen, kl_single)]."""
    if not acc:
        return {"n": 0, "gen": float("nan"), "single": float("nan")}
    g, s = np.concatenate([a for a, _ in acc]), np.concatenate([b for _, b in acc])
    return {"n": int(g.size), "gen": float(g.mean()), "single": float(s.mean())}


def reduce_with_r1(phase_jsons, r1: dict, anchor: dict | None = None, *, decisive_lead: float | None = None,
                   root: Path | None = None) -> dict:
    """Phase 6's measured dicts, in gate/rules.py's shapes:

      g5_greedy       greedy_rows_vs_ref on the greedy block   -> rules.g5_greedy
      g5_r1           {arm: {"anchor": {"n", "mean_kl_r1_runner",
                      "mean_kl_r1_single"}, "parity": R-parity (from the block),
                      K8 also "greedy": R-greedy (the runner's tokens against R1)}}
      g5_r1_mutants   {26: {"parity": ...}}                    -> rules.g5_r1
      g5_bp           {arm: {B: {"B", "first_wave": D, "mid_run": D,
                      "admitted_mid_run", "max_live"}}}, D = {"n",
                      "mean_kl_r1_batched", "mean_kl_r1_single"}
      g5_bp_mutants   {27: P at K8, B = 8; 22 too in tiny mode} -> rules.g5_bp

    r1: ref_drivers.r1_rows_pass on anchor_sequences(phase_jsons) (checked:
    the same sequences). Every KL is common.kl_rows with R1 first; every
    log-prob file is loaded only after its sha256 matches the record."""
    blocks = find_blocks(phase_jsons)
    anc = anchor_sequences(phase_jsons)
    if anchor is not None and list(anchor["keys"]) != anc["keys"]:
        raise G5DataError("the anchor given is not the one these phase JSONs define")
    lead = float(_th(None)["G5"]["greedy_decisive_lead_nats"] if decisive_lead is None else decisive_lead)
    rr = _R1Rows(r1, anc)
    uses = {(u["block"], u["seq"]): u for u in anc["uses"]}
    root = Path(root or logprob_root())
    out = {"schema": REDUCE_SCHEMA, "g5_greedy": None, "g5_r1": {}, "g5_r1_mutants": {}, "g5_bp": {},
           "g5_bp_mutants": {}, "anchor": {k: anc[k] for k in ("n_sequences", "n_rows", "n_tokens")},
           "decisive_lead_nats": lead, "files_loaded": 0}

    def put(d: dict, key, value, what: str):
        if key in d:
            raise G5DataError(f"two {what} blocks for {key!r}")
        d[key] = value

    for b in blocks:
        check, arm, mutant = b["check"], b["arm"], b["mutant"]
        if not b["needs_r1"]:  # mutant 26: R-parity only, reduced when generated
            put(out["g5_r1_mutants"], mutant, {"arm": arm, "parity": b["parity"], "block": b["id"]}, "G5-R1 mutant")
            continue
        r1_by_seq = {s["seq"]: (rr.rows(uses[(b["id"], s["seq"])], f"{b['id']} {s['seq']}") if s["tokens"] else None)
                     for s in b["sequences"]}
        if check == "greedy":
            items = [(s["seq"], r1_by_seq[s["seq"]], s["tokens"], s["finish_reason"]) for s in b["sequences"] if s["tokens"]]
            res = g5.greedy_rows_vs_ref(items, lead)
            if out["g5_greedy"] is not None:
                raise G5DataError("two greedy blocks")
            out["g5_greedy"] = {**res, "block": b["id"]}
            continue
        kind = GEN_KIND[check]
        acc = {k: [] for k in SUBSETS + ("other", "all")}
        for s in b["sequences"]:
            ref = r1_by_seq[s["seq"]]
            gen_lp = load_logprobs(s["files"][kind], root)
            single = load_logprobs(s["files"][SINGLE], root)
            out["files_loaded"] += 2
            pair = (common.kl_rows(ref, gen_lp), common.kl_rows(ref, single))
            acc["all"].append(pair)
            acc[s["label"] if s["label"] in SUBSETS else "other"].append(pair)
        if check == "g5_r1":
            a = _anchor_d(acc["all"])
            entry = {"anchor": {"n": a["n"], "mean_kl_r1_runner": a["gen"], "mean_kl_r1_single": a["single"]},
                     "parity": b["parity"], "block": b["id"]}
            if arm == "K8":
                items = [(s["seq"], r1_by_seq[s["seq"]], s["tokens"], s["finish_reason"]) for s in b["sequences"]
                         if s["tokens"]]
                g = g5.greedy_rows_vs_ref(items, lead)
                entry["greedy"] = {**g, "n": g["n_positions"]}
            put(out["g5_r1"], arm, entry, "G5-R1")
        else:  # g5_bp
            subs = {}
            for k in SUBSETS + ("other",):
                d = _anchor_d(acc[k])
                subs[k] = {"n": d["n"], "mean_kl_r1_batched": d["gen"], "mean_kl_r1_single": d["single"]}
            p = {"B": b["B"], "first_wave": subs["first_wave"], "mid_run": subs["mid_run"],
                 "admitted_mid_run": b["trace"]["admitted_mid_run"], "max_live": b["trace"]["max_live"],
                 "other": subs["other"], "descriptive_single_vs_batched": b.get("single_vs_batched"),
                 "block": b["id"]}
            if mutant is not None:
                put(out["g5_bp_mutants"], mutant, {**p, "arm": arm}, "G5-BP-lean mutant")
            else:
                put(out["g5_bp"].setdefault(arm, {}), int(b["B"]), p, f"G5-BP-lean {arm}")
    return out


# ---------------------------------------------------------------------------
# G5-D32 (§3.12)
# ---------------------------------------------------------------------------


def d32_inputs(ts, th: dict | None = None) -> tuple[dict, dict]:
    """(texts, ranges) of G5-D32 for a TextSet: the ranges th["G5D32"] names
    (T1, T3, T9) at the profile's decode positions (hi inclusive)."""
    names = _th(th)["G5D32"]["ranges"]
    texts = {"T9": ts.t9, **{t: ts.ids[t] for t in ts.ids}}
    return {r: texts[r] for r in names}, {r: tuple(ts.profile.decode_ranges[r]) for r in names}


def _comparison(base: np.ndarray, other: np.ndarray) -> dict:
    """KL(base||other) per position (common.kl_rows) and base's lead where the
    top-1 differs: R = {"n", "mean_kl", "max_kl", "top1_change_leads",
    "n_top1_changes"}."""
    kl = common.kl_rows(base, other)
    changed = base.argmax(axis=-1) != other.argmax(axis=-1)
    leads = common.top_lead(base[changed]) if changed.any() else np.zeros(0)
    return {"n": int(kl.size), "mean_kl": float(kl.mean()), "max_kl": float(kl.max()),
            "top1_change_leads": tail(leads), "n_top1_changes": int(changed.sum())}


def g5_d32(model, module, texts: dict, ranges: dict, *, chunk: int | None = None, chunk_small: int | None = None,
           comparisons=("decode", "chunk64"), th: dict | None = None, log=print) -> dict:
    """G5-D32's series (§3.12) for the K8 fp32 port (`module`: its port
    module, where ForcedTrunk installs): {range: {"positions", "n", "csort":
    {"n", "mean", "max", "max_share"}, "decode": R, "chunk64": R}}, the shape
    rules.g5_d32 reads. comparisons=("decode",) for mutant 20."""
    from gate.harness import ForcedTrunk, decode_spans, forced_logits, num_layers_of, records_table

    t = _th(th)["G5D32"]
    chunk = int(t["chunk"] if chunk is None else chunk)
    small = int(t["chunk_small"] if chunk_small is None else chunk_small)
    bad = set(comparisons) - {"decode", "chunk64"}
    if bad:
        raise ValueError(f"unknown G5-D32 comparisons {sorted(bad)}")
    L = num_layers_of(model)
    out = {}
    for name, (lo, hi) in ranges.items():
        ids = np.asarray(texts[name][:hi + 1], dtype=np.int64)
        n = ids.size
        if n != hi + 1 or not 0 <= lo <= hi:
            raise ValueError(f"G5-D32 {name}: range [{lo}, {hi}] outside a text of {len(texts[name])} tokens")
        pos, rows = np.arange(lo, hi + 1), np.arange(n)
        free_ft = ForcedTrunk(module, L, "free", shadow=False)
        free = forced_logits(model, free_ft, rows, ids, chunk, positions=pos)  # (i)
        table = records_table(free_ft.records(), n)

        def forced(spans):
            return forced_logits(model, ForcedTrunk(module, L, "force", table=table, shadow=False), rows, ids, spans,
                                 positions=pos)

        base = forced(chunk)  # (ii)
        cs = common.kl_rows(free, base)
        tot = float(cs.sum())
        entry = {"positions": [int(lo), int(hi)], "n": int(pos.size), "chunk": chunk, "chunk_small": small,
                 "csort": {"n": int(cs.size), "mean": float(cs.mean()), "max": float(cs.max()),
                           "max_share": float(cs.max() / tot) if tot > 0 else None}}
        if "decode" in comparisons:
            entry["decode"] = _comparison(base, forced(decode_spans(n, int(lo), chunk)))  # (iii)
        if "chunk64" in comparisons:
            entry["chunk64"] = _comparison(base, forced(small))  # (iv)
        out[name] = entry
        log(f"[gate] G5-D32 {name} [{lo}, {hi}]: Csort mean {entry['csort']['mean']:.3e}")
    return out
