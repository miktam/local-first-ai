# SPDX-License-Identifier: MIT
"""H1 speed class and the descriptive speed cells (HYPOTHESIS H1; BUILD_SPEC §5.7).

H1 cell ("speed", results/bench/speed_<UTC>.jsonl):
- K4 and G4 both resident, one untimed warm-up each;
- 10 blocks; in each block the order of K4 and G4 comes from one seed-36
  permutation stream (recorded in the header);
- one run = the first 1,024 tokens of the model's own tokenisation of
  pad_4k.txt (real name, hard fail if missing), fresh cache, greedy, EOS
  masked by a logits processor, exactly 256 generated tokens, through
  mlx_lm.stream_generate; mlx-lm's generation_tps (excludes prefill) and
  prompt_tps are recorded;
- 30 s idle after every run; thermalState, power source and power mode
  before and after every block;
- a block record carries ln(tps(K4) / tps(G4)). The t-test, CI and verdict
  are analysis/verdicts.py's (H1 rule), never computed here.
- Descriptive, after the 10 blocks: prefill tok/s at exactly 4,096 tokens for
  K4 and G4, 5 reps each (H1 "Descriptive": "Prefill tok/s at 4,096 tokens
  (ratio)").

Descriptive cell ("speed_desc", results/bench/speed_desc_<UTC>.jsonl), one
arm resident at a time, K8, G8, Q36-8, Q36-4, Q38-8, Q38-4, 5 reps each:
prefill tok/s at 4,096 tokens; batch-1 decode by the H1 protocol; aggregate
greedy decode at B in {1, 2, 4} through BatchGenerator (item 27; the kit's
one construction, runner.generate.make_batch_generator), one distinct
1,024-token prompt per row. A Kolibri arm runs only the batch sizes in
allowed_B ∩ {1, 2, 4}, allowed_B being the gate record's (exp_037 DESIGN
§3.14, §6.3); its "arm" record carries allowed_B and the sizes it ran.

Fixture note: pad_4k.txt holds only 3,916 Kolibri / 3,992 Gemma 4 / 3,994 Qwen
tokens, fewer than 4,096. The 4,096-token prefill prompt and the batch rows
are therefore taken from exact-length prefixes of pad_120k.txt, which starts
with the whole of pad_4k.txt (the convention D1 and E6 use for 32k-120k).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import common, genutil  # noqa: E402

SPEC_H1 = "HYPOTHESIS H1; BUILD_SPEC §5.7 speed.py"
SPEC_DESC = "HYPOTHESIS H1 Descriptive; BUILD_SPEC §5.7 speed.py (descriptive)"

H1_ARMS = ("K4", "G4")
H1_BLOCKS = 10
H1_ORDER_SEED = 36  # BUILD_SPEC §5.7: "10 blocks with randomised order (seed 36)"
H1_PROMPT_FIXTURE = "pad_4k.txt"
H1_PROMPT_TOKENS = 1024
H1_GEN_TOKENS = 256
IDLE_S = 30  # HYPOTHESIS H1: 30 s idle between runs

PREFILL_FIXTURE = "pad_120k.txt"  # see the fixture note above
PREFILL_TOKENS = 4096
PREFILL_REPS = 5

DESC_ARMS = ("K8", "G8", "Q36-8", "Q36-4", "Q38-8", "Q38-4")
DESC_REPS = 5
DESC_BATCH = (1, 2, 4)
BATCH_FIXTURE = "pad_120k.txt"

Loader = Callable[[str], tuple]


def block_orders(n_blocks: int = H1_BLOCKS, seed: int = H1_ORDER_SEED, arms=H1_ARMS) -> list[list[str]]:
    """The run order of every block: one numpy PCG64(seed) stream, one
    permutation of the arms per block."""
    rng = np.random.default_rng(seed)
    return [[arms[int(i)] for i in rng.permutation(len(arms))] for _ in range(n_blocks)]


def _prefill(model, tok, ids, eos) -> dict:
    """Prefill throughput: mlx-lm prompt_tps of a one-token generation."""
    r = genutil.timed_generate(model, tok, ids, 1, eos)
    return {k: r[k] for k in ("t_start", "t_end", "prompt_tokens", "prompt_tps", "peak_memory_gb_mlxlm", "prefill_step_size")}


def _arm_info(arm: str, d: Path, eos: list[int]) -> dict:
    return {
        "model_dir": common.redact_path(d),
        "fingerprint": common.model_fingerprint(arm, d),
        "eos_ids": eos,
    }


def run_h1(
    results_dir: Optional[Path] = None,
    *,
    loader: Loader = genutil.load_arm,
    n_blocks: int = H1_BLOCKS,
    idle_s: float = IDLE_S,
    prompt_tokens: int = H1_PROMPT_TOKENS,
    gen_tokens: int = H1_GEN_TOKENS,
    prefill_tokens: int = PREFILL_TOKENS,
    prefill_reps: int = PREFILL_REPS,
    cool: Optional[dict] = None,
) -> Path:
    """Run the H1 cell and return the path of its JSONL file."""
    results_dir = Path(results_dir or common.default_results_dir())
    common.require_identity()
    common.require_gates(H1_ARMS)

    loaded: dict[str, dict] = {}
    arms_hdr: dict[str, dict] = {}
    for arm in H1_ARMS:  # both resident for the whole cell
        model, tok, d = loader(arm)
        eos = genutil.eos_ids(d, tok)
        prompt, pinfo = genutil.fixture_ids(tok, H1_PROMPT_FIXTURE, prompt_tokens)
        pre, preinfo = genutil.fixture_ids(tok, PREFILL_FIXTURE, prefill_tokens)
        loaded[arm] = {"model": model, "tok": tok, "eos": eos, "prompt": prompt, "pre": pre}
        arms_hdr[arm] = {**_arm_info(arm, d, eos), "prompt": pinfo, "prefill_prompt": preinfo}

    orders = block_orders(n_blocks)
    header = {
        **common.base_header("speed", SPEC_H1, results_dir),
        "arms": arms_hdr,
        "block_orders": orders,
        "cool_down": cool,
        "memory_after_load": genutil.memory(),
        "protocol": {
            "arms": list(H1_ARMS),
            "blocks": n_blocks,
            "order_seed": H1_ORDER_SEED,
            "prompt_fixture": H1_PROMPT_FIXTURE,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens,
            "decoding": "greedy, EOS masked by a logits processor, fresh cache, mlx_lm.stream_generate",
            "idle_s_after_each_run": idle_s,
            "statistic": "ln(generation_tps K4 / generation_tps G4) per block",
            "prefill_descriptive": {"fixture": PREFILL_FIXTURE, "tokens": prefill_tokens, "reps": prefill_reps},
        },
    }
    path = common.new_output_path(results_dir, "speed")
    with common.JsonlCell(path, header) as w:
        for arm in H1_ARMS:
            m = loaded[arm]
            r = genutil.timed_generate(m["model"], m["tok"], m["prompt"], gen_tokens, m["eos"])
            w.append({"kind": "warmup", "arm": arm, **r})
        w.append({"kind": "idle", **common.idle(idle_s)})

        log_ratios = []
        for b, order in enumerate(orders):
            before = common.snapshot()
            tps = {}
            for pos, arm in enumerate(order):
                m = loaded[arm]
                r = genutil.timed_generate(m["model"], m["tok"], m["prompt"], gen_tokens, m["eos"])
                w.append({"kind": "run", "block": b, "position": pos, "arm": arm, **r})
                tps[arm] = r["generation_tps"]
                common.idle(idle_s)
            after = common.snapshot()
            lr = math.log(tps["K4"] / tps["G4"])
            log_ratios.append(lr)
            w.append(
                {
                    "kind": "block",
                    "block": b,
                    "order": order,
                    "generation_tps": tps,
                    "log_ratio_k4_g4": lr,
                    "before": before,
                    "after": after,
                }
            )

        prefill = {arm: [] for arm in H1_ARMS}
        for rep in range(prefill_reps):
            for arm in H1_ARMS:
                m = loaded[arm]
                before = common.snapshot()
                r = _prefill(m["model"], m["tok"], m["pre"], m["eos"])
                prefill[arm].append(r["prompt_tps"])
                w.append({"kind": "prefill", "rep": rep, "arm": arm, "before": before, **r})
                common.idle(idle_s)

        mean_lr = float(np.mean(log_ratios))
        w.finish(
            {
                "n_blocks": len(log_ratios),
                "log_ratios_k4_g4": log_ratios,
                "mean_log_ratio": mean_lr,
                "geometric_mean_ratio": math.exp(mean_lr),
                "prefill_tps_median": {a: float(np.median(v)) for a, v in prefill.items() if v},
                "note": "descriptive; the H1 verdict is computed by analysis/verdicts.py",
            }
        )
    m = None  # drop the last reference to a model before releasing memory
    loaded.clear()
    genutil.release()
    return path


def desc_batch_sizes(batch_sizes, allowed) -> tuple[int, ...]:
    """The aggregate-decode batch sizes an arm runs: all of batch_sizes for a
    peer (allowed None), allowed_B ∩ batch_sizes for a Kolibri arm (exp_037
    DESIGN §6.3), in the order of batch_sizes."""
    return tuple(int(B) for B in batch_sizes if allowed is None or int(B) in allowed)


def batch_prompts(tok, B_max: int, prompt_tokens: int) -> tuple[list[list[int]], dict]:
    """B_max distinct, consecutive prompt_tokens-long windows of the
    model's tokenisation of pad_120k.txt."""
    ids, info = genutil.fixture_ids(tok, BATCH_FIXTURE, B_max * prompt_tokens)
    rows = [ids[i * prompt_tokens : (i + 1) * prompt_tokens] for i in range(B_max)]
    info["rows_ids_sha256"] = [common.sha256_ids(r) for r in rows]
    return rows, info


def run_desc(
    results_dir: Optional[Path] = None,
    *,
    loader: Loader = genutil.load_arm,
    arms=DESC_ARMS,
    reps: int = DESC_REPS,
    idle_s: float = IDLE_S,
    prompt_tokens: int = H1_PROMPT_TOKENS,
    gen_tokens: int = H1_GEN_TOKENS,
    prefill_tokens: int = PREFILL_TOKENS,
    batch_sizes=DESC_BATCH,
    cool: Optional[dict] = None,
) -> Path:
    """Run the descriptive speed cell; one arm resident at a time. A Kolibri
    arm's aggregate decode runs only at B in allowed_B ∩ batch_sizes."""
    results_dir = Path(results_dir or common.default_results_dir())
    common.require_identity()
    gates = common.require_gates(arms)
    allowed = {arm: common.allowed_B(arm, gates.get(arm)) for arm in arms}  # None for a peer
    arm_batch = {arm: desc_batch_sizes(batch_sizes, allowed[arm]) for arm in arms}

    header = {
        **common.base_header("speed_desc", SPEC_DESC, results_dir),
        "cool_down": cool,
        "protocol": {
            "arms": list(arms),
            "reps": reps,
            "prefill": {"fixture": PREFILL_FIXTURE, "tokens": prefill_tokens},
            "decode_b1": {
                "fixture": H1_PROMPT_FIXTURE,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": gen_tokens,
                "path": "mlx_lm.stream_generate, greedy, EOS masked (H1 protocol)",
            },
            "batch": {
                "B": list(batch_sizes),
                "fixture": BATCH_FIXTURE,
                "prompt_tokens_per_row": prompt_tokens,
                "generated_tokens_per_row": gen_tokens,
                "path": "mlx_lm BatchGenerator as BUILD_SPEC item 27 (runner.generate.make_batch_generator), "
                        "greedy with EOS masked in the sampler",
                "kolibri_rule": "a Kolibri arm runs only B in allowed_B ∩ B (the gate record's; exp_037 DESIGN §6.3)",
                "B_by_arm": {arm: list(arm_batch[arm]) for arm in arms},
                "allowed_B": {arm: list(allowed[arm]) for arm in arms if allowed[arm] is not None},
            },
            "idle_s_after_each_rep": idle_s,
        },
    }
    path = common.new_output_path(results_dir, "speed_desc")
    summary: dict[str, dict] = {}
    with common.JsonlCell(path, header) as w:
        for arm in arms:
            model, tok, d = loader(arm)
            eos = genutil.eos_ids(d, tok)
            prompt, pinfo = genutil.fixture_ids(tok, H1_PROMPT_FIXTURE, prompt_tokens)
            pre, preinfo = genutil.fixture_ids(tok, PREFILL_FIXTURE, prefill_tokens)
            rows, binfo = batch_prompts(tok, max(batch_sizes), prompt_tokens)
            w.append(
                {
                    "kind": "arm",
                    "arm": arm,
                    **_arm_info(arm, d, eos),
                    "prompt": pinfo,
                    "prefill_prompt": preinfo,
                    "batch_prompts": binfo,
                    "batch_B": list(arm_batch[arm]),
                    "allowed_B": None if allowed[arm] is None else list(allowed[arm]),
                    "memory_after_load": genutil.memory(),
                }
            )
            r = genutil.timed_generate(model, tok, prompt, gen_tokens, eos)
            w.append({"kind": "warmup", "arm": arm, **r})
            common.idle(idle_s)

            acc: dict[str, list] = {"prefill_tps": [], "decode_b1_tps": []}
            for B in arm_batch[arm]:
                acc[f"aggregate_tps_B{B}"] = []
            for rep in range(reps):
                before = common.snapshot()
                p = _prefill(model, tok, pre, eos)
                w.append({"kind": "prefill", "arm": arm, "rep": rep, **p})
                acc["prefill_tps"].append(p["prompt_tps"])
                r = genutil.timed_generate(model, tok, prompt, gen_tokens, eos)
                w.append({"kind": "decode_b1", "arm": arm, "rep": rep, **r})
                acc["decode_b1_tps"].append(r["generation_tps"])
                for B in arm_batch[arm]:
                    g = genutil.batch_aggregate(model, rows[:B], gen_tokens, eos, B)
                    w.append({"kind": "batch", "arm": arm, "rep": rep, **g})
                    acc[f"aggregate_tps_B{B}"].append(g["generation_tps"])
                after = common.snapshot()
                w.append({"kind": "rep", "arm": arm, "rep": rep, "before": before, "after": after})
                common.idle(idle_s)
            summary[arm] = {k: float(np.median(v)) for k, v in acc.items() if v}
            del model, tok
            genutil.release()
        w.finish({"median_by_arm": summary, "note": "descriptive only"})
    return path
