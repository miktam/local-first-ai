"""Figure for 'The Silicon Wager' (exp_007, with exp_008's flags-off Mini)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lfa_viz as V  # noqa: E402
from _runtime import prefill_seconds  # noqa: E402

POST = "2026-05-29-silicon-wager.md"
E7 = "exp_007_hardware_comparison/evidence/"
E8 = "exp_008_flash_attention/evidence/"


def figures(repo: Path) -> dict:
    on = prefill_seconds(repo, E7 + "20260529T130052Z-phase_a-mini", E7 + "20260529T134341Z-phase_b-mini")
    off = prefill_seconds(repo, E8 + "20260607T101221Z-phase_a-mini-nf", E8 + "20260607T102637Z-phase_b-mini-nf")
    mbp = [p for p in prefill_seconds(repo, E7 + "20260529T184636Z-phase_a-mbp", E7 + "20260529T191226Z-phase_b-mbp") if p[0] <= 41000]
    fig = V.line_chart(
        [("Mac mini, flash attention on + q8_0 KV (exp_007)", "--s2", on, {"end": f"mini, flags on: {on[-1][1]:,.0f} s"}),
         ("Mac mini, flash attention off, fp16 KV (exp_008)", "--s1", off, {"end": f"mini, flags off: {off[-1][1]:.0f} s", "dy": -8}),
         ("MacBook Pro (exp_007; flags not recorded)", "--s3", mbp, {"end": f"MBP: {mbp[-1][1]:.0f} s", "dy": 8})],
        "Prefill seconds per call (Ollama's prompt_eval_duration, repeat 1), gemma4:26b, without model load or generation. "
        "The Mini's runs in exp_007 had flash attention and a q8_0 KV cache switched on (Incident 007-Alpha); with both off "
        "(exp_008, 7 June) the same Mini needs about 89 s at 40K. The MacBook Pro's flag state was not recorded; its runs to "
        "120K are in the record. Ollama 0.20.2 per the pre-registration; AC power.",
        "prompt tokens", "seconds", xticks=[4000, 10000, 20000, 30000, 40000], ydec=0,
        xfmt=lambda t: f"{t / 1000:.0f}K", label="The Mini's wall was mostly the flags")
    return {"silicon-prefill-seconds": fig}
