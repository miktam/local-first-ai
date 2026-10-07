# Experiment 036 — assets on the run host

*Recorded 2026-10-03 16:00 UTC. The kit (port, validation gate, runner, scorers) and the pre-registration land in this directory next.*

Kolibri is Aleph Alpha's English–German MoE (78.1B total / 3.46B active, Apache-2.0, released 2026-10-03). This experiment runs it on Andrei's MacBook Pro (Apple M5 Max, 128 GB unified), not on miktam-mini.

Everything lives under one folder on the run host, `$EXP036_MODELS`, default `~/models/exp036`. [`assets.json`](./assets.json) is the machine-readable copy of this table. The kit reads it and never downloads anything itself.

## Models

| Folder | Hugging Face repo | Revision | Size | Role |
|---|---|---|---|---|
| `Kolibri-1-BF16` | Aleph-Alpha/Kolibri-1-BF16 | `7a8f290e` | 156.2 GB | source for our MLX 8-bit / 4-bit conversions and the reference |
| `gemma-4-26b-a4b-it-8bit` | mlx-community/gemma-4-26b-a4b-it-8bit | `33c6d237` | 28.0 GB | peer, 8-bit |
| `Qwen3.6-35B-A3B-8bit` | mlx-community/Qwen3.6-35B-A3B-8bit | `e06a74e6` | 37.7 GB | peer, 8-bit |
| `Qwen3.8-27B-8bit` | mlx-community/Qwen3.8-27B-8bit | `815b83c0` | 29.5 GB | peer, 8-bit |
| `gemma-4-26b-a4b-it-4bit` | mlx-community/gemma-4-26b-a4b-it-4bit | `0d77464e` | 15.4 GB | peer, 4-bit |
| `Qwen3.6-35B-A3B-4bit` | mlx-community/Qwen3.6-35B-A3B-4bit | `38740b84` | 20.4 GB | peer, 4-bit |
| `Qwen3.8-27B-4bit` | mlx-community/Qwen3.8-27B-4bit | `10c35caa` | 16.1 GB | peer, 4-bit |

All peer builds are affine, group size 64 — the recipe our Kolibri conversions use.

## Datasets (`data/`)

| Folder | Source | Revision | Note |
|---|---|---|---|
| `data/aime26` | math-ai/aime26 | `79037aeb` | |
| `data/aime26-multilingual` | ellamind/aime26-multilingual | `3c8bc18f` | German AIME 2026 |
| `data/IFBench_test` | allenai/IFBench_test | `2e8a48de` | |
| `data/IFBench-src` | github.com/allenai/IFBench | `1c40f0c1` | deterministic checkers |
| `data/MMLU-ProX-Lite` | li-lab/MMLU-ProX-Lite | `e82aafb9` | |
| `data/MMLU-ProX` | li-lab/MMLU-ProX | `8e6106a6` | |
| `data/gpqa` | Idavidrein/gpqa | `83022cef` | gated; item text never published |
| `data/gpqa-multilingual` | ellamind/gpqa-multilingual | `bc70ca17` | gated; item text never published |

## Fetch commands (re-runnable; `hf download` resumes)

```bash
M=~/models/exp036; D=$M/data
caffeinate -i hf download Aleph-Alpha/Kolibri-1-BF16 --revision 7a8f290e7858825c3cf5e4c447ba68345de9f1d3 --local-dir $M/Kolibri-1-BF16
caffeinate -i hf download mlx-community/gemma-4-26b-a4b-it-8bit --revision 33c6d23798a0af159529890f79329206dbfbd73c --local-dir $M/gemma-4-26b-a4b-it-8bit
caffeinate -i hf download mlx-community/Qwen3.6-35B-A3B-8bit --revision e06a74e6236a60c8367e1a3214e83d8b61b637b0 --local-dir $M/Qwen3.6-35B-A3B-8bit
caffeinate -i hf download mlx-community/Qwen3.8-27B-8bit --revision 815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9 --local-dir $M/Qwen3.8-27B-8bit
caffeinate -i hf download mlx-community/gemma-4-26b-a4b-it-4bit --revision 0d77464eeb233a2da68ebf9d7dc4edaac7db956d --local-dir $M/gemma-4-26b-a4b-it-4bit
caffeinate -i hf download mlx-community/Qwen3.6-35B-A3B-4bit --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46 --local-dir $M/Qwen3.6-35B-A3B-4bit
caffeinate -i hf download mlx-community/Qwen3.8-27B-4bit --revision 10c35caafbb80f7dc6a7a432cdd11af10a6d4818 --local-dir $M/Qwen3.8-27B-4bit

hf download math-ai/aime26               --repo-type dataset --revision 79037aebdb6580008fb960d17cb21fd3099083e3 --local-dir $D/aime26
hf download ellamind/aime26-multilingual --repo-type dataset --revision 3c8bc18f6c185ed2d66803dbd394839a3917fd82 --local-dir $D/aime26-multilingual
hf download allenai/IFBench_test         --repo-type dataset --revision 2e8a48de45ff3bf41242f927254ca81b59ca3ae2 --local-dir $D/IFBench_test
hf download li-lab/MMLU-ProX-Lite        --repo-type dataset --revision e82aafb9460529687d3c7e51b401d8dd1dd309dd --local-dir $D/MMLU-ProX-Lite
hf download li-lab/MMLU-ProX             --repo-type dataset --revision 8e6106a6c6ce1c5027e66cc338143cf997b2aa09 --local-dir $D/MMLU-ProX
git clone https://github.com/allenai/IFBench $D/IFBench-src && git -C $D/IFBench-src checkout 1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d
hf download Idavidrein/gpqa              --repo-type dataset --revision 83022cefff930aea54f654c0b282e74b9eeda5c6 --local-dir $D/gpqa
hf download ellamind/gpqa-multilingual   --repo-type dataset --revision bc70ca1791afa241a98c972e126c2decaf0a367d --local-dir $D/gpqa-multilingual
```

## Added 2026-10-03 18:20 UTC (approved by Andrei for the pre-registration)

| Folder | Source | Pin | Note |
|---|---|---|---|
| `data/fineweb-2` | HuggingFaceFW/fineweb-2, file `data/deu_Latn/test/000_00000.parquet` | `af9c1333` | 104 MB; German tokenizer test (H5) and local-only gate texts. Never committed. |
| `data/RGB-src` | github.com/chen700564/RGB | `65ec39e4` | ~25 MB of data; CC BY-NC-SA 4.0 (code and data). Scores only are published. |
| `ifbench-venv` | built from `data/IFBench-src/uv.lock` | uv.lock `1c235a49` | IFBench checker environment |
| `nltk_data` | NLTK packages punkt, punkt_tab, stopwords, averaged_perceptron_tagger_eng | — | keeps IFBench scoring offline |

```bash
M=~/models/exp036; D=$M/data
# FineWeb-2: name the file positionally — `hf download --include` silently skipped files on the mini (hf 2.1.1)
hf download HuggingFaceFW/fineweb-2 data/deu_Latn/test/000_00000.parquet --repo-type dataset --revision af9c13333eb981300149d5ca60a8e9d659b276b9 --local-dir $D/fineweb-2
git clone https://github.com/chen700564/RGB $D/RGB-src && git -C $D/RGB-src checkout 65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615
(cd $D/IFBench-src && UV_PROJECT_ENVIRONMENT=$M/ifbench-venv uv sync --frozen --python 3.12)
$M/ifbench-venv/bin/python -c "import nltk; [nltk.download(p, download_dir='$M/nltk_data') for p in ['punkt','punkt_tab','stopwords','averaged_perceptron_tagger_eng']]"
```

The kit's preflight checks each of these against `assets.json` (revision, sha256 where recorded) before any run.

Disk: about 303 GB of downloads plus about 127 GB for the two Kolibri conversions, plus run outputs and the gate's working files — keep **~600 GB** free (the mbp has 2.9 TiB).

The vendor's technical report is not committed (Aleph Alpha keeps the rights to it). It is <https://aleph-alpha.com/downloads/tech-report.pdf>, read as retrieved 2026-10-03: 3,465,511 bytes, sha256 `01520e07506e67d53aebfb16b5298384940870d652409ea3943fe990f7a68b0e`.
