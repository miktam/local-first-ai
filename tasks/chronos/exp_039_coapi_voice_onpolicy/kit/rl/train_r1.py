#!/usr/bin/env python3
"""
exp_039 — R1 launcher around mlx-lm-lora 5.8.6 (run from coapi_voice/).

  python rl/train_r1.py                 # full run: one pass over data/rl/train.jsonl
  python rl/train_r1.py --iters 10      # the timed pilot (same schedule, stops at 10)
  python rl/train_r1.py --smoke         # mini plumbing check: 2 steps, tiny groups, short completions

Fixes the config alone cannot make, each found by reading the pinned package:
  1. build_lora_config() and utils.from_pretrained() both drop `lora_parameters.keys`, so LoRA
     silently goes on every linear layer incl. the MLP — 33.03 M trainable, exp_035's
     configuration, the very thing exp_039 holds fixed (seen in the first mini smoke test).
     The keys are put back at both places, and the run refuses unless exactly 11,796,480
     parameters are trainable.
  2. iters and the cosine decay length are one pass over the train prompts (ceil(n / batch));
     a pilot with --iters keeps the full-run schedule so its timing is representative.
  3. the package version is checked against the pin before anything loads.
  4. a --config YAML cannot override arguments with non-None CLI defaults (group_size,
     epsilon, temperature, max_completion_length): the first smoke test ran group 4,
     clip 1e-4, temperature 1.0. The config is passed as a dict instead and every effective
     value is checked against it before training.
  5. mlx-lm's tokenizer wrapper turns Qwen3 thinking ON unless told otherwise; it is
     defaulted off and the rendered rollout prompt is checked against serve's.
The resolved config is written next to the adapters as the record; the trainer receives it as a dict.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

import yaml

PIN = "5.8.6"
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "grpo_config.yaml"
# Qwen3-4B: hidden 2560, q 32×128, kv 8×128; r16 × (in + out) per projection × 36 layers
EXPECTED_TRAINABLE = 16 * ((2560 + 4096) + (2560 + 1024) * 2 + (4096 + 2560)) * 36


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, help="stop after N steps (pilot); schedule stays the full run's")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--data", help="prompt directory (default: the config's data)")
    a = ap.parse_args()

    if (v := importlib.metadata.version("mlx-lm-lora")) != PIN:
        sys.exit(f"mlx-lm-lora {v} installed, {PIN} pinned; refusing")
    import mlx_lm_lora.train as T

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg["model"] = str(Path(cfg["model"]).expanduser())
    if a.data:
        cfg["data"] = a.data
    keys = (cfg.get("lora_parameters") or {}).get("keys")
    if not keys:
        sys.exit("lora_parameters.keys must be set explicitly; refusing")
    n_train = sum(1 for l in (Path(cfg["data"]) / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    full = math.ceil(n_train / cfg["batch_size"])
    cfg["lr_schedule"]["arguments"][1] = full
    cfg["iters"] = a.iters or full
    if a.iters:                                    # a pilot never writes into the full run's directory
        cfg["adapter_path"] = time.strftime("runs_rl/pilot-%Y%m%d-%H%M%S")
    if a.smoke:
        cfg.update(iters=2, batch_size=1, group_size=2, max_completion_length=64, save_every=1,
                   adapter_path=time.strftime("runs_rl/smoke-%Y%m%d-%H%M%S"))
        cfg["lr_schedule"]["warmup"] = 0
        cfg["lr_schedule"]["arguments"][1] = 2
    out = Path(cfg["adapter_path"])
    if not (a.iters or a.smoke) and any(out.glob("*adapters.safetensors")):
        sys.exit(f"{out} already holds adapters from an earlier run; refusing to overwrite it")
    out.mkdir(parents=True, exist_ok=True)
    resolved = out / "config_resolved.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    orig = T.build_lora_config

    def build_lora_config(args):
        c = orig(args)
        c["keys"] = list(keys)
        print(f"[exp039] LoRA keys {c['keys']} · rank {c['rank']} · scale {c['scale']} · dropout {c['dropout']} "
              f"· layers {c['num_layers']} · expected trainable ≈ 11.80 M (q/k/v/o, r16, 36 layers)", flush=True)
        return c

    T.build_lora_config = build_lora_config

    # 5. thinking off at rollout, as at serve. mlx-lm 0.32's TokenizerWrapper.apply_chat_template
    #    passes enable_thinking=True for Qwen3 unless the caller sets it, and GRPODataset never
    #    does, so the first smoke test's rollouts all opened with <think> and ran to the cap.
    #    Default it to False, then prove on the first train row that the rendered prompt ends
    #    exactly as Ollama's think:false renders it.
    from mlx_lm.tokenizer_utils import TokenizerWrapper, load as load_tokenizer
    orig_act = TokenizerWrapper.apply_chat_template

    def apply_chat_template(self, *args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return orig_act(self, *args, **kwargs)

    TokenizerWrapper.apply_chat_template = apply_chat_template
    first = json.loads((Path(cfg["data"]) / "train.jsonl").read_text(encoding="utf-8").split("\n", 1)[0])
    tok = load_tokenizer(Path(cfg["model"]))
    text = tok.apply_chat_template([{"role": "system", "content": first["system"]},
                                    {"role": "user", "content": first["prompt"]}], add_generation_prompt=True, tokenize=False)
    if not text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
        sys.exit(f"rendered prompt does not end with the empty think block: {text[-60:]!r}; refusing")
    print("[exp039] rollout prompt ends with the empty think block (think: false), as at serve", flush=True)

    # 1b. utils.from_pretrained() rebuilds the config it passes to linear_to_lora_layers and drops
    #     `keys` a second time; inject them at that call site and count what became trainable.
    import mlx_lm_lora.utils as U
    from mlx.utils import tree_flatten
    orig_l2l = U.linear_to_lora_layers

    def linear_to_lora_layers(model, num_layers, config, **kw):
        orig_l2l(model=model, num_layers=num_layers, config={**config, "keys": list(keys)}, **kw)
        n = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        print(f"[exp039] trainable parameters {n:,} (expected {EXPECTED_TRAINABLE:,})", flush=True)
        if n != EXPECTED_TRAINABLE:
            sys.exit("trainable parameter count differs from the pre-registered adapter; refusing")

    U.linear_to_lora_layers = linear_to_lora_layers
    # 4. YAML values only fill arguments whose CLI default is None, and group_size (4), epsilon
    #    (1e-4), temperature (1.0) and max_completion_length (512) have non-None defaults, so a
    #    --config run silently trains with the package's values. main(dict) overrides every
    #    default; run() is wrapped to refuse unless each effective value equals the config.
    orig_run = T.run

    def run(args, training_callback=None):
        bad = {k: (getattr(args, k, None), v) for k, v in cfg.items() if getattr(args, k, None) != v}
        if bad:
            sys.exit(f"effective arguments differ from the config (effective, config): {bad}; refusing")
        print(f"[exp039] effective = config for all {len(cfg)} keys · group {args.group_size} · beta {args.beta} "
              f"· epsilon {args.epsilon} · temperature {args.temperature} · max_completion {args.max_completion_length}", flush=True)
        return orig_run(args, training_callback)

    T.run = run
    print(f"[exp039] n_train {n_train} · batch {cfg['batch_size']} · iters {cfg['iters']} of {full} · config {resolved}", flush=True)
    sys.argv = [sys.argv[0]]
    T.main(dict(cfg))


if __name__ == "__main__":
    main()
