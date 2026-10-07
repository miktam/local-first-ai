# SPDX-License-Identifier: MIT
"""exp_036 bench cells (BUILD_SPEC §5.7).

    fit.py              D1   K4 peak memory at 32k / 64k context
    speed.py            H1   K4 vs G4 batch-1 decode; descriptive speed cells
    tokenizer_ratio.py  H5   German bytes/token, Kolibri vs Gemma 4 vs Qwen
    kl_8v4.py           H8   KL(8-bit || 4-bit) per UTF-8 byte on T1-T6
    batch_flip.py       C1   greedy answer flips, memory-rule B vs B = 1
    ladder.py           E6   context ladder (Tier 2; frozen by its own amendment)
    run_bench.py        driver, fixed cell order, skip-if-complete
    common.py           shared plumbing: paths, arms, guards, records, thermals

Each cell writes one self-describing file (HYPOTHESIS "Evidence layout"):
results/bench/{speed,speed_desc,fit,c1,ladder}_<UTC>.jsonl,
results/tokenizer_<UTC>.json and results/kl_8v4_<UTC>.json. Verdicts are not
computed here; analysis/ reads these files.
"""
