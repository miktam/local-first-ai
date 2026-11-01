# Aborted run, 2026-11-01 06:25Z

Started at 12:26 CET by the chained block after exp_057's Amendment 2. Arm T finished (220 of 220, last row 12:47). Arm F reached 41 of 220 (last row 15:07) and was stopped by hand.

**Why it was stopped.** The mini was under a thermal clamp that was not heat. The system's thermal pressure level reached 4, the highest ("sleeping"), within about 15 minutes of load. The GPU was held at 338 to 618 MHz against a top clock near 1,580 MHz, and decoding fell from 56.5 tokens a second at T's first message to about 8.5 from F's X022 on. After the stop the level stayed at 4 with the machine idle: 36 mW drawn, every die sensor at or below 54.3 °C. The firmware (20457.41.3) was newer than the running macOS 26.5.1's loader (18000.120.36): the staged macOS 27.2 had updated the firmware without the OS. On 27.2, with firmware and loader both at 20457.41.3, the level read 0 and five minutes of gemma4:26b held 58 to 60 tokens a second at level 0.

**What is kept and why.** The answers were produced at temperature 0 with seed 55, so they stay as a record. The timings measure the clamp, not the arms, and arm T spans two machine states. Nothing in this folder is scored. The measured run starts fresh on macOS 27.2 (HYPOTHESIS, Amendment 1).

**Held back.** The thinking text quotes sender addresses from the corpus, and the model can write variants of them that were never checked against DNS. Both files are kept out of this repository, as exp_057 kept its machine log:

- `T/thinking.jsonl` sha256 `6a9a906ec9b100972a74eac909e389f0507d4a13a787652de15196babfd8a79e`
- `F/thinking.jsonl` sha256 `e6402a5e29eadb2b52d17e9ef6af19ea43d7334ba25c473e043f265c0dd2da54`
