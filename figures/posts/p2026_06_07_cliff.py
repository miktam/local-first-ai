"""Figure for 'The Cliff That Wasn't' (exp_007, exp_008, exp_010)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lfa_viz as V  # noqa: E402
from _runtime import prefill_curve  # noqa: E402

POST = "2026-06-07-the-cliff-that-wasnt.md"


def figures(repo: Path) -> dict:
    A, tA = prefill_curve(repo, "exp_008_flash_attention/evidence/20260607T101221Z-phase_a-mini-nf", "exp_008_flash_attention/evidence/20260607T102637Z-phase_b-mini-nf")
    B, tB = prefill_curve(repo, "exp_010_fa_isolation/evidence/20260607T110817Z-phase_a-mini-condB", "exp_010_fa_isolation/evidence/20260607T113235Z-phase_b-mini-condB")
    Cc, tC = prefill_curve(repo, "exp_010_fa_isolation/evidence/20260607T130322Z-phase_a-mini-condC", "exp_010_fa_isolation/evidence/20260607T140224Z-phase_b-mini-condC")
    D, tD = prefill_curve(repo, "exp_007_hardware_comparison/evidence/20260529T130052Z-phase_a-mini", "exp_007_hardware_comparison/evidence/20260529T134341Z-phase_b-mini")
    fig = V.line_chart(
        [("flash attention on, q8_0 KV (D)", "--s2", D, {"dash": True, "end": "on, q8_0"}),
         ("flash attention on, fp16 KV (B)", "--s2", B, {"end": "on, fp16"}),
         ("flash attention off, fp16 KV (A)", "--s1", A, {"end": "off, fp16 and q8_0", "dy": -6}),
         ("flash attention off, q8_0 KV (C)", "--s1", Cc, {"dash": True, "end": None})],
        "Prefill time per token on the Mac mini (gemma4:26b, Ollama, num_ctx 131072, temperature 0), one cold call per size "
        "(repeat 1, after a fresh unload). Both flash-attention-on lines climb; both off lines stay near 2 ms/tok through 40K. "
        f"The thin lines are the cliff thresholds of the two 'on' conditions (2× their own 15K value: B {tB:.2f}, D {tD:.2f} ms/tok). "
        "D was measured on 29 May (exp_007); A, B and C on 7 June (exp_008, exp_010).",
        "prompt tokens", "ms per token", xticks=[4000, 10000, 20000, 30000, 40000], ydec=0,
        hrefs=[(tB, f"threshold B {tB:.1f}", "--s2"), (tD, f"threshold D {tD:.1f}", "--s2")],
        xfmt=lambda t: f"{t / 1000:.0f}K", label="The cliff follows the flash-attention flag")
    return {"cliff-fa-2x2": fig}
