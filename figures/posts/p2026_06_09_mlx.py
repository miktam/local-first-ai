"""Figure for 'MLX vs Ollama' (exp_011 against exp_008 and exp_010)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lfa_viz as V  # noqa: E402
from _runtime import prefill_curve  # noqa: E402

POST = "2026-06-09-mlx-vs-ollama-runtime.md"


def figures(repo: Path) -> dict:
    M, tM = prefill_curve(repo, "exp_011_mlx_runtime/evidence/20260609T064038Z-phase_a-mini", "exp_011_mlx_runtime/evidence/20260609T065708Z-phase_b-mini")
    A, _ = prefill_curve(repo, "exp_008_flash_attention/evidence/20260607T101221Z-phase_a-mini-nf", "exp_008_flash_attention/evidence/20260607T102637Z-phase_b-mini-nf")
    Cc, _ = prefill_curve(repo, "exp_010_fa_isolation/evidence/20260607T130322Z-phase_a-mini-condC", "exp_010_fa_isolation/evidence/20260607T140224Z-phase_b-mini-condC")
    fig = V.line_chart(
        [("MLX (OptiQ 4-bit)", "--s3", M, {"end": "MLX", "dy": -14}),
         ("Ollama, flash attention off, fp16 KV", "--s1", A, {"end": "Ollama, fp16", "dy": -6}),
         ("Ollama, flash attention off, q8_0 KV", "--s1", Cc, {"dash": True, "end": "Ollama, q8_0", "dy": 8})],
        "Prefill time per token on the Mac mini, gemma4:26b, one cold call per size (repeat 1). The prompts differ: the MLX "
        "script missed its fixture files and fed one sentence, repeated and cut short of each target (35,555 tokens at the 40K "
        f"setting), while Ollama read the real fixtures. x is the actual prompt length. The thin line is MLX's own cliff threshold ({tM:.1f} ms/tok); "
        "MLX stays under it to the end. The quantisation differs too (OptiQ 4-bit against GGUF Q4_K_M), so this is not a pure "
        "runtime comparison; mlx-lm's version is not recorded.",
        "prompt tokens", "ms per token", xticks=[4000, 10000, 20000, 30000, 40000], ymax=4, ydec=0,
        hrefs=[(tM, f"MLX threshold {tM:.1f}", "--s3")], xfmt=lambda t: f"{t / 1000:.0f}K", label="MLX against Ollama, prefill")
    return {"mlx-prefill": fig}
