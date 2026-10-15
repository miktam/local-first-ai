# exp_039 brief: COAPI Voice, on-policy — form and grounding from a verifier, not from a teacher

*Written 2026-10-12 after Andrei said he wants to train again, "using what Coinbase did and my previous experience". Design in [DESIGN.md](DESIGN.md). Nothing is built, nothing has run, no value exists.*

**What it tests.** exp_035 trained the voice three times by SFT on 363 answers written by a stronger model and lost substance every time, with the cause left as a hypothesis (the data-and-objective; the confound that every target was Fable's prose). Coinbase (Oct 2026) post-trained a 9B with GRPO against a deterministic outcome reward and gained. exp_039 changes exactly one thing against exp_035's v2 arm: **the objective**. Same base (Qwen3-4B), same retrieval and prompts, same glue as the control, no stronger-model answers anywhere in the loop. The model's own samples are scored by code — the checks `run_eval.py` already makes, plus arithmetic prompts whose answers are computed in code — and the Fable judge stays outside the reward, used once on sealed sets.

**Arms.** B0 untrained 4B, no glue · A12 untrained 4B + glue (the reference) · R1 the GRPO-trained 4B, no glue · R1+G the trained 4B behind the same glue (the candidate).

**Sets.** v1 (60 rows) for the substance question, because both compared arms run without glue and the model never saw its questions. v2 (30 fresh-rule rows, built 2026-09-23, **still unsealed**) for form and the ceiling items, scored once per arm. Sealing v2 is step 0 and it is also exp_035's overdue re-score.

**Hypotheses (thresholds in DESIGN §4).**
- **H2a, the one that matters:** R1 keeps the base's substance (with-substance correctness ≥ B0 − 0.10 on v1, paired; harmful rows ≤ B0's). exp_035's SFT lost 0.3.
- **H1:** R1 meets ≥ 6 of 7 format floors on v2 with no post-processing (SFT did 6/7; the base 1/7).
- **H3:** R1+G moves at least one thing the glue cannot fix: Spanish term at first use, tail collapse, harmful rows.
- **H5, a gate:** a blind read of 40 top-reward rollouts finds ≤ 4 gamed ones. More → STOP, the verifier is unsound.
- Verdict map, in order: Stopped (verifier) → No gain (substance lost under RL too; the post's claim widens) → Substance kept, form not learned → Form from the verifier, nothing over glue → Earns the phone trial.

**Cost.** Build ~1.5 days (reward module + tests, prompt builder, mlx-lm-lora config, mini smoke test). Andrei ~3 h (seal review, decisions, the mbp run by hand). mbp wall time capped at 12 h by a frozen rule after a 10-step pilot; no cloud GPU unless the cap is hit. Judge: about 410 cloud judgments through this session's subagents, never the API. Mini: ~1 h of scoring.

**When.** Step 0 (seal v2, score A12 on it) can happen today and stands on its own. Build starts after decisions 1–4. The run waits for Andrei's sign-off on the pre-registration.

**Decisions (recommendation in bold)**
1. Platform: **the mbp with mlx-lm-lora 5.8.6 (GRPO on MLX, custom reward file), smoke-tested on the mini first** / one rented NVIDIA GPU with TRL + vLLM as fallback. *(Changed 2026-10-13: GRPO now exists for MLX; see PREP.md.)*
2. Visibility: **public pre-registration, reward code and prompt builder; sets, glue and guide bound by sha256 as in exp_038.**
3. Training prompt: **the glue's language-rendered system prompt, so train = serve** / exp_035's fixed English prompt (a purer test of weights, and the thing that pulled ES/PL answers into English).
4. KL leash: **β = 0.02** / 0 (TRL's default; Coinbase doesn't say).
5. Seal eval v2 first: **yes, now.**
6. The scope-gate decision head (latent-space item 23): **out; one variable at a time; it is exp_040 if the harness works.**
7. Stop rules: **pilot projects > 12 h on the mbp → STOP; verifier audit > 4/40 gamed → STOP; both as written.**
