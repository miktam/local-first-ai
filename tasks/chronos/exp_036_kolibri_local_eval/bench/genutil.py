# SPDX-License-Identifier: MIT
"""MLX helpers for the bench cells: loading an arm, EOS masking, timed
single-sequence generation through mlx-lm's own stream_generate, and batched
generation through mlx-lm's BatchGenerator built as BUILD_SPEC item 27 says.

mlx and mlx_lm are imported inside the functions, after bench.common has set
HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from bench import common

PREFILL_STEP = 2048  # item 27 and D1: prefill_step_size 2048


def load_arm(arm: str, model_dir: Optional[Path] = None):
    """(model, tokenizer, model_dir) for an arm, through mlx_lm.load.

    Kolibri builds run their own copy of kolibri1.py; it must equal port/
    (port/README.md, INTEGRATION_LOG R3), so check_port_file runs first."""
    d = Path(model_dir) if model_dir else common.arm_dir(arm)
    if not d.is_dir():
        raise common.BenchError(f"{arm}: model directory {common.redact_path(d)} is missing")
    if common.is_kolibri(arm):
        from port.convert import check_port_file

        check_port_file(d)
    from mlx_lm import load

    model, tokenizer = load(str(d))
    return model, tokenizer, d


def release() -> None:
    """Drop unreferenced arrays and MLX's buffer cache between models."""
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def eos_ids(model_dir: Path, tokenizer: Any = None) -> list[int]:
    """Every id that ends generation: generation_config.json eos_token_id,
    config.json eos_token_id and the loaded tokenizer's eos_token_ids (the set
    stream_generate stops on)."""
    ids: set[int] = set()
    for name in ("generation_config.json", "config.json"):
        p = Path(model_dir) / name
        if p.is_file():
            v = json.loads(p.read_text()).get("eos_token_id")
            if isinstance(v, int):
                ids.add(v)
            elif isinstance(v, list):
                ids.update(int(x) for x in v)
    if tokenizer is not None:
        ids.update(int(x) for x in (getattr(tokenizer, "eos_token_ids", None) or []))
    if not ids:
        raise common.BenchError(f"no EOS ids found for {common.redact_path(model_dir)}")
    return sorted(ids)


def _vocab_mask(ids: Sequence[int], vocab: int):
    import mlx.core as mx

    m = np.zeros(vocab, dtype=bool)
    m[[i for i in ids if 0 <= i < vocab]] = True
    return mx.array(m)


class EosMask:
    """Logits processor that sets every EOS logit to -inf (BUILD_SPEC §5.7 H1:
    "EOS masked (a logits processor setting EOS to -inf)")."""

    def __init__(self, ids: Sequence[int]):
        self.ids = sorted(set(int(i) for i in ids))
        self._masks: dict[int, Any] = {}

    def __call__(self, tokens, logits):
        import mlx.core as mx

        v = logits.shape[-1]
        if v not in self._masks:
            self._masks[v] = _vocab_mask(self.ids, v)
        return mx.where(self._masks[v], -mx.inf, logits)


def greedy(logprobs):
    import mlx.core as mx

    return mx.argmax(logprobs, axis=-1)


def masked_greedy(ids: Sequence[int]) -> Callable:
    """Greedy sampler that never picks an EOS id; one vectorised op per step,
    used for the batched descriptive cells instead of per-row processors."""
    import mlx.core as mx

    masks: dict[int, Any] = {}

    def sampler(logprobs):
        v = logprobs.shape[-1]
        if v not in masks:
            masks[v] = _vocab_mask(ids, v)
        return mx.argmax(mx.where(masks[v], -mx.inf, logprobs), axis=-1)

    return sampler


def encode(tokenizer: Any, text: str) -> list[int]:
    """The model's own tokenisation of plain text: no special tokens, no
    template (H1 "the model's own tokenisation of pad_4k.txt")."""
    return [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]


def fixture_ids(tokenizer: Any, name: str, n: Optional[int] = None) -> tuple[list[int], dict]:
    """The first n tokens (or all) of a padding fixture in this tokenizer.
    Fewer than n tokens is a hard failure, never a silent shorter prompt."""
    text, sha = common.read_fixture(name)
    ids = encode(tokenizer, text)
    total = len(ids)
    if n is not None:
        if total < n:
            raise common.BenchError(f"{name} has {total} tokens in this tokenizer; {n} are needed")
        ids = ids[:n]
    return ids, {
        "fixture": name,
        "fixture_sha256": sha,
        "fixture_tokens": total,
        "n_tokens": len(ids),
        "ids_sha256": common.sha256_ids(ids),
    }


def memory() -> dict:
    import mlx.core as mx

    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def timed_generate(
    model,
    tokenizer,
    prompt_ids: Sequence[int],
    max_tokens: int,
    eos: Sequence[int],
    prefill_step_size: int = PREFILL_STEP,
    mask_eos: bool = True,
) -> dict:
    """One fresh-cache greedy run through mlx_lm.stream_generate.

    Returns mlx-lm's own prompt_tps and generation_tps (which excludes
    prefill; H1). With mask_eos the run must produce exactly max_tokens
    tokens and finish by length, otherwise BenchError."""
    import mlx.core as mx
    from mlx_lm.generate import stream_generate

    processors = [EosMask(eos)] if mask_eos else None
    tokens: list[int] = []
    last = None
    t0 = common.utc_iso()
    for r in stream_generate(
        model,
        tokenizer,
        mx.array(list(prompt_ids), dtype=mx.int32),
        max_tokens=max_tokens,
        sampler=greedy,
        logits_processors=processors,
        prefill_step_size=prefill_step_size,
    ):
        tokens.append(int(r.token))
        last = r
    t1 = common.utc_iso()
    if last is None:
        raise common.BenchError("stream_generate produced nothing")
    out = {
        "t_start": t0,
        "t_end": t1,
        "prompt_tokens": int(last.prompt_tokens),
        "prompt_tps": float(last.prompt_tps),
        "generation_tokens": int(last.generation_tokens),
        "generation_tps": float(last.generation_tps),
        "peak_memory_gb_mlxlm": float(last.peak_memory),
        "finish_reason": last.finish_reason,
        "completion_ids_sha256": common.sha256_ids(tokens),
        "eos_masked": bool(mask_eos),
        "prefill_step_size": int(prefill_step_size),
    }
    if mask_eos and (out["generation_tokens"] != max_tokens or out["finish_reason"] != "length" or len(tokens) != max_tokens):
        raise common.BenchError(
            f"expected exactly {max_tokens} tokens with EOS masked, got "
            f"{out['generation_tokens']} ({out['finish_reason']})"
        )
    for k in ("prompt_tps", "generation_tps"):
        if not math.isfinite(out[k]):
            raise common.BenchError(f"{k} is not finite")
    return out


def _batch_generator(model, B: int, max_tokens: int, stop_tokens, sampler):
    """BatchGenerator exactly as BUILD_SPEC item 27 constructs it."""
    from mlx_lm.generate import BatchGenerator

    return BatchGenerator(
        model,
        max_tokens=max_tokens,
        stop_tokens=stop_tokens,
        sampler=sampler,
        completion_batch_size=B,
        prefill_batch_size=min(B, 8),
        prefill_step_size=PREFILL_STEP,
        max_kv_size=None,
    )


def _drain(gen, uids: list[int]) -> tuple[dict[int, list[int]], dict[int, str]]:
    toks: dict[int, list[int]] = {u: [] for u in uids}
    fin: dict[int, str] = {}
    empty = 0
    while len(fin) < len(uids):
        responses = gen.next_generated()
        if not responses:
            empty += 1
            if empty > 1000:
                raise common.BenchError("BatchGenerator stalled with unfinished sequences")
            continue
        empty = 0
        for r in responses:
            toks[r.uid].append(int(r.token))
            if r.finish_reason is not None:
                fin[r.uid] = r.finish_reason
    return toks, fin


def batch_aggregate(model, prompts: list[list[int]], max_tokens: int, eos: Sequence[int], B: int) -> dict:
    """Aggregate greedy decode throughput with B live sequences (descriptive;
    HYPOTHESIS H1 "aggregate decode at B in {1, 2, 4}"). One distinct prompt
    per row, EOS masked, exactly max_tokens per row. generation_tps is
    BatchGenerator.stats(): all generated tokens over the wall time minus the
    prompt-processing time."""
    if len(prompts) != B:
        raise common.BenchError(f"batch_aggregate needs exactly B={B} prompts")
    gen = _batch_generator(model, B, max_tokens, None, masked_greedy(eos))
    t0 = common.utc_iso()
    try:
        with gen.stats() as st:
            uids = gen.insert([list(p) for p in prompts], max_tokens=[max_tokens] * B)
            toks, fin = _drain(gen, uids)
    finally:
        gen.close()
    counts = [len(toks[u]) for u in uids]
    if any(c != max_tokens for c in counts) or any(fin[u] != "length" for u in uids):
        raise common.BenchError(f"batched run produced {counts} tokens, expected {max_tokens} each")
    return {
        "t_start": t0,
        "t_end": common.utc_iso(),
        "B": B,
        "prompt_tokens": int(st.prompt_tokens),
        "prompt_tps": float(st.prompt_tps),
        "generation_tokens": int(st.generation_tokens),
        "generation_tps": float(st.generation_tps),
        "peak_memory_gb_mlxlm": float(st.peak_memory),
        "prompt_ids_sha256": [common.sha256_ids(p) for p in prompts],
        "completion_ids_sha256": [common.sha256_ids(toks[u]) for u in uids],
    }


def batch_completions(
    model,
    prompts: list[list[int]],
    B: int,
    max_tokens: int,
    stop_ids: Sequence[int],
    sampler: Optional[Callable] = None,
) -> list[dict]:
    """Greedy completions of every prompt with at most B live sequences,
    inserted in the given order and refilled as sequences finish (item 27).
    The stop token, when one ends a sequence, is the last completion id."""
    gen = _batch_generator(model, B, max_tokens, [[int(i)] for i in stop_ids], sampler or greedy)
    try:
        uids = gen.insert([list(p) for p in prompts], max_tokens=[max_tokens] * len(prompts))
        toks, fin = _drain(gen, uids)
    finally:
        gen.close()
    return [{"completion_ids": toks[u], "finish_reason": fin[u]} for u in uids]
