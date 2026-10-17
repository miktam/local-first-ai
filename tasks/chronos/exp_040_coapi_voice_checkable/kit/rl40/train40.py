#!/usr/bin/env python3
"""
exp_040 — R3 launcher: exp_039's rl/train_r1.py unchanged, pointed at rl40/grpo_config40.yaml (run from coapi_voice/).

  python rl40/train40.py                 # full run: one pass over data/rl40/train/train.jsonl (78 steps)
  python rl40/train40.py --iters 10      # the timed pilot
  python rl40/train40.py --smoke         # plumbing check: 2 steps, tiny groups, short completions

train_r1 carries exp_039's five guards (LoRA keys put back and the trainable count checked at 11,796,480; one-pass
schedule; package pin 5.8.6; config passed as a dict and every effective value checked; thinking off at rollout,
checked against serve's rendering). Its [exp039] log prefix is kept: the code is exp_039's, byte-identical.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rl"))
import train_r1  # noqa: E402

train_r1.CONFIG = HERE / "grpo_config40.yaml"   # read as main()'s --config default

if __name__ == "__main__":
    train_r1.main()
