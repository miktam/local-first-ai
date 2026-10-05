"""G5 first-wave A/B on the real K8 (mbp). Investigation only: no gate record, nothing under
results/ or diagnostics/gate1/out, aggregates only (no token ids, no text).

Arms (the gate's G5 batch profile, B = 8, greedy, the gate's single comparator logits_at chunk 2048):
  A  the gate's construction (gate/checks/g5_generation._batch_generator): must reproduce
     mean KL 0.3073773544462904 exactly
  B  as A, but mlx_lm right-pads the first-wave prefill with another token id (1 instead of 0).
     In exact arithmetic, and bitwise on the M4 Pro, the valid rows cannot depend on the pad
     content. Any difference from A = a kernel that mixes rows of the padded batch.
  C  as A, but prefill_batch_size = 1: every prompt is prefilled alone (B = 1, no right
     padding, like the mid-run admissions) and merged into the B = 8 decode batch.
  D  as A, but the 8 first-wave prompts all have the longest length (no right padding,
     distinct content, B = 8 prefill).

usage: python g5_ab_mbp.py --out $EXP036_WORK/bughunt_g5/ab_<utc>.json [--arms ABCD] [--tiny ROOT]
"""
import argparse, importlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault("MLX_ENABLE_TF32", "0")
KIT = Path(os.environ.get("EXP036_KIT", "$KIT"))
sys.path[:0] = [str(KIT)]
from tools.precision import ensure_exact_fp32
PREC = ensure_exact_fp32()
import numpy as np
import mlx.core as mx
from gate import common, run_gate
from gate.checks import g5_generation as g5
from gate.harness import logits_at

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--arms", default="ABCD"); ap.add_argument("--tiny", default=None)
args = ap.parse_args()
out_path = Path(args.out).expanduser().resolve()
if KIT.resolve() in out_path.parents:
    raise SystemExit("refusing to write inside the kit")
th = common.load_thresholds()
if args.tiny:
    root = Path(args.tiny).expanduser().resolve()
    ctx = run_gate.GateContext(models_dir=root, work_dir=root / "work", thresholds=th, tiny=True,
                               results_dir=root / "results", head_policy="quantised_head", subprocess_phases=False)
else:
    ctx = run_gate.GateContext(models_dir=common.models_dir(), work_dir=common.work_dir(), thresholds=th,
                               subprocess_phases=False)
ts = run_gate.textset_for(ctx)
prof = ts.profile
B = th["G5"]["batch_B"]
ORDER = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
LEN = list(prof.batch_lengths); MT = list(prof.batch_max_tokens)
prompts_A = [ts.ids[ORDER[j % 8]][: LEN[j]] for j in range(len(LEN))]
d = ctx.arm_dir("K8")
if not args.tiny:
    from port.convert import check_port_file
    check_port_file(d)
model, _, _ = common.load_port(d)
mx.eval(model.parameters())
GEN = importlib.import_module("mlx_lm.generate")
RECORD = 0.3073773544462904


def loop(prompts, prefill_bs=None, pad_id=None):
    orig = GEN._right_pad_prompts
    if pad_id is not None:
        GEN._right_pad_prompts = lambda p, max_length=None: mx.array(
            [q + [pad_id] * ((max_length or max(len(r) for r in p)) - len(q)) for q in p])
    try:
        gen, _ = g5._batch_generator(model, B, common.EOS_IDS, max(MT))
        if prefill_bs is not None:
            gen.prefill_batch_size = prefill_bs
        got, queue, uid_of, in_flight, step, any_finished, mid = {}, list(range(len(prompts))), {}, 0, 0, False, {}
        try:
            while queue or in_flight:
                free = B - in_flight
                if free > 0 and queue:
                    group = queue[:free]; del queue[:free]
                    uids = gen.insert([list(prompts[j]) for j in group], max_tokens=[int(MT[j]) for j in group])
                    for u, j in zip(uids, group):
                        uid_of[u] = j; got[u] = {"tokens": [], "lp": []}; mid[j] = any_finished
                    in_flight += len(group)
                _, resps = gen.next()
                for r in resps:
                    g = got[r.uid]; g["tokens"].append(int(r.token))
                    x = r.logprobs.astype(mx.float32); mx.eval(x); g["lp"].append(np.array(x))
                    if r.finish_reason is not None:
                        in_flight -= 1; any_finished = True
                step += 1
        finally:
            gen.close()
    finally:
        GEN._right_pad_prompts = orig
    seqs = {}
    for u, j in uid_of.items():
        g = got[u]; p = prompts[j]; P = len(p)
        seqs[j] = {"tokens": g["tokens"], "lp": np.stack(g["lp"]), "P": P, "mid": mid[j]}
    return seqs


def score(prompts, seqs):
    rows, kls = [], []
    for j in sorted(seqs):
        s = seqs[j]; P, n = s["P"], len(s["tokens"])
        single = g5._lp(logits_at(model, list(prompts[j]) + s["tokens"][:-1], np.arange(P - 1, P - 1 + n)))
        k = g5.kl_lp(single, g5._lp(s["lp"]))
        dis = single.argmax(-1) != s["lp"].argmax(-1)
        kls.append(k)
        for t in range(n):
            rows.append((j, t, s["mid"], float(k[t]), bool(dis[t])))
    allk = np.concatenate(kls)
    def m(sel):
        v = [r[3] for r in rows if sel(r)]
        return {"n": len(v), "mean_kl": float(np.mean(v)) if v else None}
    return {"mean_kl": float(allk.mean()), "top1_dis": float(np.mean([r[4] for r in rows])), "n_positions": int(allk.size),
            "classes": {"first_wave": m(lambda r: not r[2]), "mid_run": m(lambda r: r[2]), "t0": m(lambda r: r[1] == 0),
                        "t1_2": m(lambda r: 1 <= r[1] <= 2), "t3plus": m(lambda r: r[1] >= 3),
                        "fw_t0": m(lambda r: not r[2] and r[1] == 0), "fw_t1_2": m(lambda r: not r[2] and 1 <= r[1] <= 2),
                        "fw_t3plus": m(lambda r: not r[2] and r[1] >= 3), "mid_t0": m(lambda r: r[2] and r[1] == 0),
                        "mid_t1plus": m(lambda r: r[2] and r[1] >= 1)},
            "per_seq": [{"j": j, "P": seqs[j]["P"], "n": len(seqs[j]["tokens"]), "mid_run": seqs[j]["mid"],
                         "mean_kl": float(np.mean([r[3] for r in rows if r[0] == j])),
                         "max_kl": float(np.max([r[3] for r in rows if r[0] == j]))} for j in sorted(seqs)]}


def compare(sa, sb):
    """Per sequence: first token where the greedy tokens differ, and whether the batched
    log-probs are bitwise equal on the common prefix (up to and including that step)."""
    out = []
    for j in sorted(sa):
        ta, tb = sa[j]["tokens"], sb[j]["tokens"]
        k = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), min(len(ta), len(tb)))
        upto = min(k + 1, len(ta), len(tb))
        la, lb = sa[j]["lp"][:upto], sb[j]["lp"][:upto]
        out.append({"j": j, "mid_run": sa[j]["mid"], "first_token_divergence": k if k < min(len(ta), len(tb)) else None,
                    "lp_bitwise_equal_on_common_prefix": bool(np.array_equal(la, lb)),
                    "max_abs_dlogprob_common_prefix": float(np.abs(la - lb).max()),
                    "first_step_dlogprob_nonzero": next((i for i in range(upto) if not np.array_equal(la[i], lb[i])), None)})
    return out


res = {"what": "bughunt g5-bf16 A/B (investigation, not a gate run)", "device": mx.device_info().get("device_name"),
       "mlx": mx.__version__, "precision": PREC, "tiny": bool(args.tiny), "B": B, "batch_lengths": LEN, "batch_max_tokens": MT,
       "arms": {}}
seqsA = None
for arm in args.arms:
    t0 = time.time()
    if arm == "A":
        prompts = prompts_A; seqs = loop(prompts); seqsA = seqs
    elif arm == "B":
        prompts = prompts_A; seqs = loop(prompts, pad_id=1)
    elif arm == "C":
        prompts = prompts_A; seqs = loop(prompts, prefill_bs=1)
    elif arm == "D":
        Lmax = max(LEN[:B])
        prompts = [ts.ids[ORDER[j % 8]][: (Lmax if j < B else LEN[j])] for j in range(len(LEN))]
        seqs = loop(prompts)
    else:
        raise SystemExit(arm)
    r = score(prompts, seqs)
    if arm == "A" and not args.tiny:
        r["reproduces_record"] = r["mean_kl"] == RECORD
    if arm in "BC" and seqsA is not None:
        r["vs_A"] = compare(seqsA, seqs)
    r["wall_s"] = time.time() - t0
    res["arms"][arm] = r
    c = r["classes"]
    print(f"[{arm}] mean_kl {r['mean_kl']:.6f} first_wave {c['first_wave']['mean_kl']:.4f} mid_run {c['mid_run']['mean_kl']} "
          f"fw_t0 {c['fw_t0']['mean_kl']} fw_t1_2 {c['fw_t1_2']['mean_kl']} ({r['wall_s']:.0f}s)", flush=True)
    if "vs_A" in r:
        print(f"[{arm}] vs A: first-wave rows bitwise equal on common prefix: "
              f"{[x['lp_bitwise_equal_on_common_prefix'] for x in r['vs_A'] if not x['mid_run']]}", flush=True)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(res, indent=1))
print("wrote", out_path)
