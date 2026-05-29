#!/usr/bin/env python3
"""
Exp 007 — Padding fixture generator.

Generates synthetic English-text padding files at target token counts.
Actual token count is confirmed by prompt_eval_count during benchmarking;
these fixtures are pre-committed and never modified after first use.

Usage:
    python3 generate_padding.py          # generate all sizes
    python3 generate_padding.py --verify # print expected char counts only

Target token → char conversion: 4 chars ≈ 1 token (Gemma tokenizer, English prose).
"""

import argparse
import textwrap
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "fixtures" / "padding"

# Varied neutral English sentences. Long enough to avoid trivial repetition patterns.
CORPUS = [
    "The throughput of a distributed system depends on the balance between latency at each node and the capacity of the network connecting them.",
    "When measuring prefill performance on unified memory architectures, the key bottleneck is memory bandwidth rather than raw compute cycles per second.",
    "A well-designed cache hierarchy reduces redundant data fetches and improves the effective utilisation of available memory bandwidth on the system.",
    "Transformer models use self-attention to relate all positions within a sequence simultaneously, which is powerful but computationally quadratic in input length.",
    "The operational envelope of a language model on consumer hardware is defined by the point where memory bandwidth becomes the binding constraint on prefill speed.",
    "Local inference offers significant privacy advantages over cloud-hosted models, since no data leaves the machine during any phase of the computation.",
    "Real estate data in coastal Spain often includes discrepancies between cadastral records and notarial surface area measurements, requiring reconciliation at the transaction stage.",
    "Thermal throttling on a mobile processor can reduce sustained performance by up to thirty percent compared to the initial burst throughput measured at the start of a workload.",
    "Evidence-based infrastructure decisions require pre-registered hypotheses, controlled sweep designs, and append-only result directories that cannot be retroactively altered.",
    "The Apple M4 Pro chip integrates CPU, GPU, and unified memory on a single die, which eliminates the PCIe bandwidth bottleneck common in discrete GPU configurations.",
    "Generating long-context responses on Apple Silicon requires careful management of the key-value cache, whose size grows linearly with sequence length and is stored in unified memory.",
    "Cascade architectures pair a small routing model with a larger synthesis model to reduce total inference cost while preserving answer quality on well-structured queries.",
    "GDPR compliance in AI-assisted real estate operations requires that personal data about owners, buyers, and legal proceedings never appear in structured property output fields.",
    "A pre-registered experiment assigns pass and failure criteria before collecting any data, which prevents post-hoc rationalisation of ambiguous or unfavourable results.",
    "Memory bandwidth saturation on the M4 Pro manifests as simultaneously declining GPU and CPU utilisation, indicating that both processors are waiting for data rather than computing.",
    "Ollama manages model loading, context window allocation, and KV cache eviction on behalf of the calling application, exposing a REST interface that abstracts these details.",
    "The difference between a useful benchmark and a misleading one often lies in whether the fixture sizes span the regime of interest, including the region beyond the performance cliff.",
    "Synthesising health data from an Apple Watch corpus requires aggregating millions of raw sample records into monthly or yearly statistics before they fit within a local model's context.",
    "Router models can classify user queries into tractable sub-problems and route each to the appropriate data slice, reducing the bundle size sent to the larger synthesis model.",
    "A sustained ninety-minute generation workload on a laptop differs from the same workload on a desktop due to the laptop's thermal management and battery-discharge constraints.",
    "The Marbella real estate market is characterised by a high proportion of international buyers, significant price variance between urbanisations, and opaque listing data from multiple agencies.",
    "Benchmarking across two machines requires careful controls: identical model weights, identical runtime versions, identical environment variables, and AC power throughout the measurement window.",
    "The prefill phase of transformer inference involves processing all input tokens in parallel to populate the key-value cache, after which autoregressive decoding generates output tokens one at a time.",
    "Incident analysis in engineering organisations benefits from iterative revision: the initial framing of a failure mode is often refined as more evidence accumulates and simpler explanations are tested.",
    "Property surface area discrepancies in Spain can arise from historical measurement differences, rounding in the Catastro registry, or undocumented modifications made after the original registration.",
    "A data sovereignty architecture ensures that proprietary business data, such as client negotiations and internal pricing intelligence, never reaches third-party model providers for training or inference.",
    "Fencing as a sport produces a distinctive Apple Watch workout signature: high heart-rate variability, short explosive bouts, and session lengths that align with standard competitive bout structures.",
    "The verifiability contract for AI infrastructure claims requires that every empirical assertion link to a replayable artefact with a timestamp, an environment specification, and raw result files.",
    "When a model's KV cache overflows available GPU memory, the runtime must either spill to CPU memory, truncate the context, or fail the request — each option carrying a different performance cost.",
    "Effective anonymisation in a local memory system rewrites the model's working world-model during inference, not just the input and output strings, which is a stronger privacy guarantee than surface-level filtering.",
]

SIZES = {
    "pad_4k":    4_000,
    "pad_8k":    8_000,
    "pad_15k":  15_000,
    "pad_20k":  20_000,
    "pad_22500": 22_500,
    "pad_25k":  25_000,
    "pad_27500": 27_500,
    "pad_30k":  30_000,
    "pad_32500": 32_500,
    "pad_35k":  35_000,
    "pad_37500": 37_500,
    "pad_40k":  40_000,
}

CHARS_PER_TOKEN = 4


def generate_text(target_tokens: int) -> str:
    target_chars = target_tokens * CHARS_PER_TOKEN
    sentences = CORPUS.copy()
    lines = []
    chars = 0
    idx = 0
    while chars < target_chars:
        sentence = sentences[idx % len(sentences)]
        line = sentence + " "
        lines.append(line)
        chars += len(line)
        idx += 1
    text = " ".join(s.strip() for s in lines)
    return text[:target_chars]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify", action="store_true",
                   help="Print expected char counts without writing files")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nExp 007 — Padding fixture generator")
    print(f"Output: {OUT_DIR}\n")
    print(f"{'Name':12}  {'Target tok':12}  {'Target chars':14}  {'Actual chars':14}  {'Status'}")
    print("-" * 70)

    for name, target_tok in SIZES.items():
        target_chars = target_tok * CHARS_PER_TOKEN
        text = generate_text(target_tok)
        actual_chars = len(text)
        out_path = OUT_DIR / f"{name}.txt"
        status = "verify-only" if args.verify else "written"
        if not args.verify:
            out_path.write_text(text, encoding="utf-8")
        print(f"{name:12}  {target_tok:12,}  {target_chars:14,}  {actual_chars:14,}  {status}")

    if not args.verify:
        print(f"\nAll fixtures written to {OUT_DIR}/")
        print("Confirm actual token counts via prompt_eval_count in Phase A results.\n")


if __name__ == "__main__":
    main()
