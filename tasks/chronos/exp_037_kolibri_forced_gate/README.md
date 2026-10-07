# Experiment 037 — Kolibri-1 through a forced gate

Aleph Alpha's Kolibri-1 (78.1B total / 3.46B active MoE, English and German), run through our MLX port on an Apple M5 Max next to Gemma 4 and Qwen3.6 / Qwen3.8. exp_037 re-registers the hypotheses of [exp_036](../exp_036_kolibri_local_eval/), whose port-correctness gate failed, with a gate that forces the routing so that a correct port can pass it, and that shows each new check's power in the same run. Status and verdicts live in [HYPOTHESIS.md](./HYPOTHESIS.md); nothing here is a result.

| Document | What it is |
|---|---|
| [HYPOTHESIS.md](./HYPOTHESIS.md) | The pre-registration: hypotheses H1–H8 and D1, the Phase 0 gate, decision rules, arms, task sets, sessions, the decisions record, the disclosures and the hash table of everything frozen. It wins over every other file here. |
| [BUILD_SPEC.md](./BUILD_SPEC.md) | The kit, as a delta over exp_036's build specification. |
| [RUNBOOK.md](./RUNBOOK.md) | The steps Andrei and the mbp session follow, in order. |
| [BUILD_LOG.md](./BUILD_LOG.md) | How the kit was built and tested on the mini, task by task. |
| [ASSETS.md](./ASSETS.md) · [assets.json](./assets.json) | Pinned models, datasets and code (exp_036's; exp_037 adds none). |
| [COPY_RECORD.json](./COPY_RECORD.json) | Every file of the kit: those copied from exp_036, with their source and copy sha256, and those new in exp_037, each with its status and owner task. |
| [NOTICE](./NOTICE) · [LICENSE-APACHE-2.0](./LICENSE-APACHE-2.0) | The Apache-2.0 files in this directory; everything else is MIT. |

| Directory | Contents |
|---|---|
| `port/` | The MLX model file (`kolibri1.py`, exp037-port-1: vLLM-angle RoPE, attention hooks) and the converter (exp_037 makes no conversion; it clones exp_036's builds). |
| `reference/` | The numpy fp32 reference forward pass, its mutants and derivation table (unchanged from exp_036). |
| `gate/` | Phase 0: the checks (measurement only), `rules.py` (every decision), frozen thresholds, controls and the F_tiny calibration, the port mutants, the gate texts. |
| `runner/` | Batched generation (one BatchGenerator construction), the vLLM-order sampler, chat templates, memory rule, allowed_B, plan fixing. |
| `tasks/` | Loaders, prompt renderers, the vendored eval-framework files, and exp_036's item manifests (reused). |
| `scorers/` | Deterministic scorers: extractors, RGB rule, IFBench adapter. |
| `bench/` | Speed (H1), fit (D1), tokenizer (H5), 8-bit vs 4-bit KL (H8), batch flip (C1). |
| `analysis/` | Statistics, the verdict code and the H2 tripwire. |
| `tools/` | Preflight, version record, leak check and its pre-push hook, tree hashes, peer check, build refresh, status, KV bytes, redaction, and the descriptive power log (`power_log.py`: powermetrics' CPU + GPU + ANE power joined to the run's windows, written to `results/power/`; RUNBOOK's "Power log" steps 9–12, 14 and 16; it decides nothing, and the raw `.plist` logs are never committed). |
| `env/` | Exact pins (`requirements-mbp.txt`, `versions.json`), `exp037.settings.sh`, `setup.sh`. exp_036's env file is sourced in place and never copied. |
| `diagnostics/` | The Gemma fidelity producer (a copy of exp_036's), the [tiny stress records](./diagnostics/tiny_stress/README.md) and the [vendor-code spike](./diagnostics/vendor_spike/README.md). |
| `tests/` | Weight-free unit tests and the whole gate on tiny checkpoints; they pass on the mini before any push. |
| `host/` | The run host's preflight records from before exp_036's kit existed. |

Before any commit or push: `tools/leak_check.py --range @{u}..HEAD --staged` (installed as the pre-push hook from `tools/hooks/pre-push`). No withheld item text, gold or raw output (GPQA, RGB, AIME-DE) and no FineWeb text is ever committed.
