# exp_051: MLX against Ollama on the same text

This experiment re-runs exp_011's comparison. exp_011's MLX script never loaded its fixture files, so it measured a repeated "quick brown fox" sentence. Registered and run on 2026-10-24 on the Mac mini (M4 Pro, 64 GB).

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the pre-registration (arms, protocol, H1–H6, decision table) and Amendment 1 |
| [RESULT.md](RESULT.md) | verdicts, the numbers, what they change, limits |
| `kit/` | `bench51.py` (one arm per run), `score51.py` (the registered scorer), `run51.sh` (the whole run in order), `build_natural.py` |
| `natural/` | the natural-text inputs, built from exp_043's, exp_044's and exp_049's published synthetic enquiries; `MANIFEST.json` holds their hashes |
| `runs/20261024T085524Z/` | the scored run: one folder per arm with every call (`calls.jsonl`) and its summary, `run.log`, `score.json` |
| `aborted/20261024T085417Z/` | the first launch, which made no model call (see its `NOTE.md`) |

**Re-running.** The kit is published as it ran, and two of its paths are specific to our machine:
- It reads exp_007's fixtures from `~/REPOS/local-first-ai/tasks/chronos/exp_007_hardware_comparison/fixtures/padding`. Clone the repository there, or edit `FIX` in `bench51.py` (and `C` in `build_natural.py`).
- It reads the Ollama log at `/opt/homebrew/var/log/ollama.log` to detect outside requests. That is the log path of a Homebrew Ollama.

Run order: `HF_HUB_OFFLINE=1 zsh kit/run51.sh` from this folder.

**Versions:**
- Ollama 0.40.2, `gemma4:26b` (`777b927e3b6b`)
- mlx 0.32.4 and mlx-lm 0.32.2, model `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit` (snapshot `dbfd2a77`)
- Python 3.14, macOS 26.5.1
