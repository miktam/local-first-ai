# exp_065, dev: shortlist of models not yet on the disk (2026-11-27)

Web only; nothing was downloaded. Gathered by a research session and spot-checked by the dispatcher. It feeds the registration's "at most two larger models"; the choice is Andrei's.

**Rules applied:** weights from the original maker, no build made to remove refusals; a GGUF from the maker, ggml-org, unsloth or bartowski; an architecture supported by a recent llama.cpp release; a file of at most about 85 GB for the 128 GB machine and about 40 GB for the 64 GB machine; a licence that allows local use.

**Ranking:** the Artificial Analysis Intelligence Index (II), as shown on its model pages on 2026-11-27. Those scores were presumably not measured on 4-bit files. For comparison, Qwen3.8 27B, already on the disk, is at II 20.

| Model (maker) | Released | Size | II | 4-bit file | Fits 64 GB | Licence |
|---|---|---|---|---|---|---|
| Ling-3.2-air (InclusionAI) | 2026-11-23 | 124B, mixture of experts, 5.5B active | 23 | bartowski Q4_K_M 80.4 GB (three parts), Q4_0 72.8 GB | no | MIT |
| Nanbeige4.3-33B (Nanbeige Lab) | 2026-10-25 | 33B | 20 | bartowski Q4_K_M 20.8 GB | yes | Apache-2.0 |
| K2-Horizon-32B (IFM) | 2026-09-03 | 32B | 14 | bartowski Q4_K_M 22.0 GB | yes | Apache-2.0 |
| Nemotron 3.5 Super VL 120B-A12B (NVIDIA) | 2026-10-15 | 120B, mixture of experts | 14 | ggml-org Q4_0 72.1 GB | no | OpenMDW-1.1 |
| K2-Horizon-MoVA-36B-A4B (IFM) | 2026-09-03 | 36B, mixture of experts | 14 | bartowski Q4_K_M 22.7 GB | yes | Apache-2.0 |

**Checked by the dispatcher on Hugging Face's API the same day:** `inclusionAI/Ling-3.2-air` and `bartowski/Ling-3.2-air-GGUF` exist (created 2026-11-23, MIT; Q4_K_M parts of 39.5, 39.8 and 1.1 GB; Q4_0 parts of 39.6 and 33.2 GB); `Nanbeige/Nanbeige4.3-33B` and `bartowski/Nanbeige_Nanbeige4.3-33B-GGUF` exist (created 2026-10-24 and 25, Apache-2.0; Q4_K_M 20.8 GB, sha256 beginning `2cef79855ca47ee6`).

**Not verified:**

- that any of the five returns tool calls through `llama-server`; no report was found for any of them;
- that they load on `b12676` (the Ling quants were built on `b12600`, the Nanbeige ones on `b11845`);
- the index scores themselves, which are a third party's;
- nothing in any model card describes how the model handles security questions, except NVIDIA's, which lists security training sets and gives no refusal figures.

**Looked at and left out:** Qwen3.8-Flash-Next (177B) and Inkling 2 Small (594B) are too large; Ornith-1.5-122B fits, but every score is its maker's own; Ripple-65B-Preview has no GGUF from a listed source.

Sources: artificialanalysis.ai/models/open-source and the model pages there; huggingface.co/inclusionAI/Ling-3.2-air; huggingface.co/bartowski/Ling-3.2-air-GGUF; huggingface.co/Nanbeige/Nanbeige4.3-33B; huggingface.co/bartowski/Nanbeige_Nanbeige4.3-33B-GGUF; huggingface.co/IFM/K2-Horizon-32B; huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B; huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF; github.com/ggml-org/llama.cpp/pull/26608; github.com/ggml-org/llama.cpp/pull/29535.
