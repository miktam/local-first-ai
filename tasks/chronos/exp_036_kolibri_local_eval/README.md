# Experiment 036 — Kolibri-1 evaluated locally

Aleph Alpha's Kolibri-1 (78.1B total / 3.46B active MoE, English and German), ported to MLX, checked against a separately written numpy reference, and run on an Apple M5 Max next to Gemma 4 and Qwen3.6 / Qwen3.8. Status and verdicts live in [HYPOTHESIS.md](./HYPOTHESIS.md); nothing here is a result.

| Document | What it is |
|---|---|
| [HYPOTHESIS.md](./HYPOTHESIS.md) | The pre-registration: hypotheses H1–H8 and D1, decision rules, arms, task sets, sessions, the hash table of everything frozen. It wins over every other file here. |
| [BUILD_SPEC.md](./BUILD_SPEC.md) | The kit, file by file. |
| [RUNBOOK.md](./RUNBOOK.md) | The steps Andrei and the mbp session follow, in order. |
| [ASSETS.md](./ASSETS.md) · [assets.json](./assets.json) | Pinned models, datasets and code, and the fetch commands. |
| [NOTICE](./NOTICE) · [LICENSE-APACHE-2.0](./LICENSE-APACHE-2.0) | The Apache-2.0 files in this directory; everything else is MIT. |

| Directory | Contents |
|---|---|
| `port/` | The MLX model file (`kolibri1.py`) and the BF16 → MLX 8/4-bit converter. |
| `reference/` | The numpy fp32 reference forward pass, its mutants and derivation table. |
| `gate/` | Phase 0 port-correctness gate: checks, frozen thresholds, gate texts. |
| `runner/` | Batched generation, the vLLM-order sampler, chat templates, memory rule, plan fixing. |
| `tasks/` | Loaders, prompt renderers, the vendored eval-framework files, manifests (built on the run host). |
| `scorers/` | Deterministic scorers: extractors, RGB rule, IFBench adapter. |
| `bench/` | Speed (H1), fit (D1), tokenizer (H5), 8-bit vs 4-bit KL (H8), batch flip (C1). |
| `analysis/` | Statistics and the verdict code. |
| `tools/` | Preflight, version record, leak check and its pre-push hook, tree hashes, peer check, status, KV bytes, redaction. |
| `env/` | Exact pins (`requirements-mbp.txt`, `versions.json`), `exp036.env`, `setup.sh`. |
| `tests/` | Weight-free unit tests; they pass on the build host before any push. |
| `host/` | The run host's preflight records from before the kit existed. |

Before any commit or push: `tools/leak_check.py --range @{u}..HEAD --staged` (installed as the pre-push hook from `tools/hooks/pre-push`). No withheld item text, gold or raw output (GPQA, RGB, AIME-DE) and no FineWeb text is ever committed.
