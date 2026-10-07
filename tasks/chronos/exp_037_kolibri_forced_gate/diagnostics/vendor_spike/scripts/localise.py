"""Localise vendor-vs-reference differences (W13 spike, diagnostics only).

Inputs: out/ref_<ck>_<seq>.npz (fp32 reference, free run) and
out/vllmAd_<ck>_<seq>.npz (vLLM + vendor plugin, with the recorded router
logits, embedding output and loaded expert bias).

Per case:
  * embedding and loaded expert_bias: vendor vs checkpoint (exact?).
  * routing: vendor's top-k (torch.topk order on logits + bias, recomputed
    from its recorded fp32 router logits) vs the reference's free-run top-k;
    every (layer, token) whose expert SET differs, with both sides' gap between
    the k-th and (k+1)-th biased score.
  * teacher-forced local error per layer l: the reference's layer l run on
    the VENDOR's input h_{l-1} (embedding for l = 0), (a) natural routing,
    (b) forced onto the vendor's selection; relative L2 against the vendor's
    h_l. Plus the reference router logits on that input vs the vendor's.
  * final norm and LM head, each on the vendor's own input.
  * chaos control (reference only): the reference's own free run restarted
    after layer 0 from h_0 plus Gaussian noise of relative L2 size 1.5e-6
    (the size of the layer-0 vendor/reference difference); relative L2 of
    every later layer against the unperturbed reference.
"""
import json
import sys

sys.dont_write_bytecode = True
E36 = str(__import__("pathlib").Path(__file__).resolve().parents[4] / "exp_036_kolibri_local_eval")  # exp_037 copy: was the absolute path of exp_036's directory on the mini
sys.path.insert(0, E36)
import numpy as np  # noqa: E402
from reference.kolibri_ref import KolibriReference  # noqa: E402

V = sys.argv[1]
cases = [c.split(":") for c in sys.argv[2].split(",")]


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def relrow(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    r = np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)
    return float(r.max()), int(r.argmax())


def topk_sets(logits, bias, k):
    s = logits.astype(np.float32) + bias.astype(np.float32)[None, :]
    order = np.argsort(-s, axis=-1, kind="stable")
    srt = np.take_along_axis(s, order, -1)
    return order[:, :k], srt[:, k - 1] - srt[:, k]


out = {}
for ck, seq in cases:
    r = np.load(f"{V}/out/ref_{ck}_{seq}.npz")
    g = np.load(f"{V}/out/vllmAd_{ck}_{seq}.npz")
    ref = KolibriReference(f"{V}/ck/{ck}")
    cfg = ref.cfg
    k = cfg.num_experts_per_tok
    ids = r["ids"]
    rep = {"ck": ck, "seq": seq}
    emb = ref.embed(ids)
    rep["embedding_exact"] = bool(np.array_equal(emb, g["emb"]))
    rep["bias_exact"] = all(np.array_equal(ref.ckpt.get(f"model.layers.{l}.moe.router.expert_bias"), g["bias"][l])
                            for l in range(cfg.num_hidden_layers))
    # reference free-run routing
    ref.forward(ids)
    ref_ids = [i for _, i in ref.last_routing]
    rows = []
    flips = []
    h_in = g["emb"]
    for l in range(cfg.num_hidden_layers):
        v_ids, v_gap = topk_sets(g["router_logits"][l], g["bias"][l], k)
        set_diff = [t for t in range(len(ids)) if set(v_ids[t]) != set(ref_ids[l][t])]
        # teacher-forced local tests on the vendor's input
        br_nat = ref.layer_branches(l, h_in)
        br_f = ref.layer_branches(l, h_in, force_ids=v_ids)
        info = br_nat["info"]
        r_ids_local, r_gap_local = topk_sets(info["logits"], ref._w(f"model.layers.{l}.moe.router.expert_bias"), k)
        local_set_diff = [t for t in range(len(ids)) if set(v_ids[t]) != set(r_ids_local[t])]
        _, r_gap_free = (None, None)
        row = {
            "layer": l, "type": cfg.layer_types[l],
            "free_rel_l2": rel(g["hidden"][l], r["hidden"][l]),
            "free_flips_tokens": len(set_diff),
            "local_nat_rel_l2": rel(g["hidden"][l], br_nat["h_out"]),
            "local_nat_rowmax": relrow(g["hidden"][l], br_nat["h_out"])[0],
            "local_forced_rel_l2": rel(g["hidden"][l], br_f["h_out"]),
            "local_forced_rowmax": relrow(g["hidden"][l], br_f["h_out"])[0],
            "local_router_rel_l2": rel(g["router_logits"][l], info["logits"]),
            "local_flips_tokens": len(local_set_diff),
        }
        for t in set_diff[:5]:
            flips.append({"layer": l, "token": t, "vendor_set": sorted(map(int, v_ids[t])),
                          "ref_set": sorted(map(int, ref_ids[l][t])), "vendor_gap": float(v_gap[t])})
        for t in local_set_diff[:5]:
            flips.append({"layer": l, "token": t, "local": True, "vendor_gap": float(v_gap[t]),
                          "ref_local_gap": float(r_gap_local[t])})
        rows.append(row)
        h_in = g["hidden"][l]
    rep["layers"] = rows
    rep["flips"] = flips
    rep["final_norm_local"] = rel(g["final"], ref.final_norm(g["hidden"][-1]))
    rep["head_local"] = rel(g["logits"], ref.lm_head(g["final"]))
    # chaos control
    rng = np.random.default_rng(7)
    h0 = r["hidden"][0].astype(np.float32)
    noise = rng.standard_normal(h0.shape).astype(np.float32)
    noise *= np.float32(1.5e-6 * np.linalg.norm(h0) / np.linalg.norm(noise))
    h = h0 + noise
    chaos = [rel(h, h0)]
    for l in range(1, cfg.num_hidden_layers):
        h = ref.layer_forward(l, h)
        chaos.append(rel(h, r["hidden"][l]))
    hn = ref.final_norm(h)
    chaos_logits = rel(ref.lm_head(hn), r["logits"])
    rep["chaos_control_rel_l2"] = chaos
    rep["chaos_control_logits_rel_l2"] = chaos_logits
    out[f"{ck}:{seq}"] = rep
    print(f"== {ck} {seq}: emb exact {rep['embedding_exact']}, bias exact {rep['bias_exact']}, "
          f"final-norm local {rep['final_norm_local']:.1e}, head local {rep['head_local']:.1e}")
    print("  l type     free     flips | local_nat rowmax  | forced   rowmax   | router  lflips | chaos")
    for row, c in zip(rows, chaos):
        print(f"  {row['layer']:<2d}{row['type'][:5]:>6s} {row['free_rel_l2']:.1e} {row['free_flips_tokens']:5d} | "
              f"{row['local_nat_rel_l2']:.1e} {row['local_nat_rowmax']:.1e} | {row['local_forced_rel_l2']:.1e} "
              f"{row['local_forced_rowmax']:.1e} | {row['local_router_rel_l2']:.1e} {row['local_flips_tokens']:5d} | {c:.1e}")
    print(f"  chaos-control logits rel_l2 {chaos_logits:.2e}; free logits rel_l2 {rel(g['logits'], r['logits']):.2e}")
    for f in flips[:12]:
        print("  flip", f)
json.dump(out, open(f"{V}/out/localise.json", "w"), indent=1)
