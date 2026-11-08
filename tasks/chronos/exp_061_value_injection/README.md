# exp_061 — does value post-training change what a 27B model does on our material?

**Complete, 2026-11-08: [RESULT.md](RESULT.md).**
H1 helps, at the threshold (all four on 36 of 40 units against the base's 32); H2 no effect (no reframing in either arm); H3 replicates (any propaganda 24.3 % → 1.4 %).

Nous Research's Arete-27B is Qwen3.8-27B post-trained by SFT to remove Chinese state-aligned answers; the vendor reports 22 % → 0.9 % on its own 803-prompt eval and no capability change on its internal bench. This experiment puts the base and the fine-tune, same quantiser and quantisation, through three things we can measure ourselves: the translation pilot's 40 blind-graded units (H1), 20 Soviet-era units chosen for political content with a "reframed" reading (H2), and 70 of the vendor's own prompts judged blind with the vendor's rubric by a different judge (H3).

Registered 2026-11-08 before any pull or draw: [HYPOTHESIS.md](HYPOTHESIS.md). Kit and runs follow under `kit/` and `runs/`; the result goes to `RESULT.md`.
