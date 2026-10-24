"""Figure for 'Our Own Gate Said No' (exp_036)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lfa_viz as V  # noqa: E402

POST = "2026-10-05-our-own-gate-said-no.md"
D = "tasks/chronos/exp_036_kolibri_local_eval/diagnostics/gate1"


def figures(repo: Path) -> dict:
    r = json.load(open(repo / D / "confirm/out/rope_chaos_20261005T145122Z.json"))["part1_chunk64_forced"]
    g = json.load(open(repo / D / "mini/bughunt/reference/gemma_chaos_16384_summary.json"))["15000-15300"]["A_vs_B_prefill64"]
    rows = [
        {"label": "Kolibri, free routing", "mean": r["p2048_vs_p64_free"]["mean_kl"], "max": r["p2048_vs_p64_free"]["max_kl"], "var": "--s2",
         "note": f"{r['p2048_vs_p64_free']['top1_changes']} top-1 flips"},
        {"label": "Kolibri, experts forced", "mean": r["p2048_vs_p64_forced"]["mean_kl"], "max": r["p2048_vs_p64_forced"]["max_kl"], "var": "--s1",
         "note": f"{r['p2048_vs_p64_forced']['top1_changes']} top-1 flips"},
        {"label": "control: forced onto itself", "mean": r["p2048_vs_p2048_forced_self"]["mean_kl"], "max": r["p2048_vs_p2048_forced_self"]["max_kl"], "var": "--ink2",
         "note": f"{r['p2048_vs_p2048_forced_self']['top1_changes']} top-1 flips"},
        {"label": "Gemma 4 26B, free routing", "mean": g["mean_kl"], "max": g["max_kl"], "var": "--s3",
         "note": f"{g['top1_changes']} top-1 flips"},
    ]
    fig = V.log_dot_range(rows,
        "KL divergence (nats) between the port's next-token distributions at positions 15,000–15,300 (n = 301), log scale, with chunk 2048 against chunk 64 in every row except the control. The "
        "dot is the mean, the line runs to the maximum (a range, not a confidence interval); one run per row. Changing only the "
        "chunk size moves Kolibri by about 0.05 nats on average; forcing the expert choices brings that to about 3e-9. The frozen "
        "bound (1e-6) applies to the maximum. Kolibri: K8 dequantised to float32, M5 Max, MLX 0.31.2. Gemma: M4 Pro, 4-bit build, "
        "its own tokenizer, so positions do not line up by content. The port against itself only; the forced comparison with the "
        "reference came in exp_037.",
        ref=1e-6, ref_label="frozen bound 1e-6", xlab="KL divergence, nats (log scale)", label="Forcing the expert choices removes the drift")
    return {"e36-forcing": fig}
