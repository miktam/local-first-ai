# Project Chronos

> *If a claim is in a Nestor post, the evidence sits in this directory.*

Chronos is the verifiability contract for [localfirstai.eu](https://localfirstai.eu).
Every empirical claim Nestor publishes — a benchmark number, an
incident, a capability — is grounded in an artefact you can open and
re-run from this directory. If the artefact is not here, the claim
should not be in a post.

This README is the index. The canonical record is
[`scientific_log.md`](./scientific_log.md).

## Layout

```
chronos/
├── scientific_log.md          # canonical log; one entry per experiment or incident
├── roadmap.md                 # forward-looking; what's planned, not yet logged
├── exp_NNN_<slug>/            # one directory per experiment (001–018)
├── incident_NNN_<slug>/       # one directory per investigated incident
├── watcher_run_NNN/           # production adversarial watcher runs
└── experiments/               # unnumbered pre-Chronos bench scripts
```

The numbering is shared between experiments and incidents because
they share scientific weight: an incident investigation that produces
preregistered hypotheses and on-disk evidence is no less rigorous
than a planned experiment.

## How to read an entry

Each `scientific_log.md` entry names artefact paths under this
directory. To verify a claim:

1. Find the entry in the log.
2. Follow its referenced paths (e.g.
   `incident_003_alpha/evidence/2026-04-28T11-10-21Z-H6/sizes.tsv`).
3. The artefact is a file you can open, parse, or — for scripts —
   re-run.

If a claim cannot be traced to an artefact under `chronos/`, treat it
as commentary, not measurement.

## Conventions

**Experiments** are planned investigations. They have a hypothesis,
a method, and a result. The directory contains source code, input
data references, and result outputs.

**Incidents** are unplanned failures investigated after the fact.
They follow a slightly different shape — a triggering event, a set
of preregistered hypotheses, scripts that test each one, and an
append-only `evidence/` directory of timestamped runs. See
[`incident_003_alpha/README.md`](./incident_003_alpha/README.md) for
the canonical structure.

**Evidence is append-only.** Once a script run produces an evidence
directory, it is not modified afterward. Re-runs create new
timestamped directories. Aborted runs move to an `aborted/`
subdirectory but are not deleted. This is what makes the log
auditable across time: a claim written today and disputed in six
months can be tested against the same on-disk evidence.

## Current contents

| Entry | Type | Status | Path |
| --- | --- | --- | --- |
| Exp 001 | Chronos activation | Complete | [`exp_001_verification_of_veracity/`](./exp_001_verification_of_veracity/) |
| Exp 002 | Control plane vs data plane (thinking tax) | Complete | [`exp_002_control_plane_vs_data_plane/`](./exp_002_control_plane_vs_data_plane/) |
| Exp 003 | Local memory anonymisation | Complete | [`exp_003_local_memory/`](./exp_003_local_memory/) |
| Incident 003-Alpha | Prefill scaling cliff | Closed | [`incident_003_alpha/`](./incident_003_alpha/) |
| Exp 004 | Bootstrap diet | Complete | [`exp_004_bootstrap_diet/`](./exp_004_bootstrap_diet/) |
| Exp 005 | Router / Reducer cascade | Complete | [`exp_005_dicer_describer/`](./exp_005_dicer_describer/) |
| Exp 006 | Redactor fidelity (GDPR, 0/20 leaks) | Complete | [`exp_006_redactor_fidelity/`](./exp_006_redactor_fidelity/) |
| Exp 007 | Mac Mini vs MacBook Pro M5 Max | Complete | [`exp_007_hardware_comparison/`](./exp_007_hardware_comparison/) |
| Exp 008 | Flash Attention + q8_0 KV cache | Complete | [`exp_008_flash_attention/`](./exp_008_flash_attention/) |
| Exp 009 | Adversarial critic (local vs frontier) | Complete | [`exp_009_adversarial_critic/`](./exp_009_adversarial_critic/) |
| Exp 010 | FA vs q8_0 factorial isolation | Complete | [`exp_010_fa_isolation/`](./exp_010_fa_isolation/) |
| Exp 011 | MLX runtime vs Ollama context cliff | Complete ⚠ MLX read a repeated sentence; re-run in Exp 051 | [`exp_011_mlx_runtime/`](./exp_011_mlx_runtime/) |
| Exp 012 | Cost vs capability curve | Complete | [`exp_012_cost_capability/`](./exp_012_cost_capability/) |
| Exp 013 | Local audit loop scaffolding | Complete | [`exp_013_local_audit_loop/`](./exp_013_local_audit_loop/) |
| Exp 014 | Capability variance floor | Complete | [`exp_014_capability_variance/`](./exp_014_capability_variance/) |
| Exp 015 | Active-parameter ablation (dense vs MoE) | Attempted, not resolved | [`exp_015_active_param_ablation/`](./exp_015_active_param_ablation/) |
| Exp 016 | Two-mac orchestration | Phase B in progress | [`exp_016_two_mac_orchestration/`](./exp_016_two_mac_orchestration/) |
| Exp 017 | Argos Phase 0 — feed reconnaissance | Pre-registered | [`exp_017_argos_phase0/`](./exp_017_argos_phase0/) |
| Exp 018 | Sovereignty resilience (3 failure modes) | Complete | [`exp_018_sovereignty_resilience/`](./exp_018_sovereignty_resilience/) |
| Exp 019 | Adversarial legal review pipeline | Complete | [`exp_019_adversarial_legal_review/`](./exp_019_adversarial_legal_review/) |
| Exp 020 | Hardening and red-teaming the inference node | Complete | [`exp_020_miktam_mini_hardening/`](./exp_020_miktam_mini_hardening/) |
| Exp 021 | Independent red-team pass (zero-escalation credential reachability) | H3 confirmed; published as accepted risk | [`exp_021_independent_red_team/`](./exp_021_independent_red_team/) |
| Exp 022 | Adversarial red-team of the CasaSol Guide bot | Complete | [`exp_022_bot_red_team/`](./exp_022_bot_red_team/) |
| Exp 023 | Generation efficiency across the local model family | Complete | [`exp_023_local_model_efficiency/`](./exp_023_local_model_efficiency/) |
| Exp 024 | Vision capability: gemma4:26b vs qwen3.8:27b | Inconclusive | [`exp_024_vision_model_comparison/`](./exp_024_vision_model_comparison/) |
| Exp 025 | Context-allocation isolation (gemma4:31b's slowdown) | H1+H2 confirmed, H3 superseded — fixed directly | [`exp_025_context_config_isolation/`](./exp_025_context_config_isolation/) |
| Exp 026 | Contextual retrieval on the COAPI corpus, fully local | Complete (pilot scale) | [`exp_026_contextual_retrieval/`](./exp_026_contextual_retrieval/) |
| Exp 036 | Kolibri on a MacBook: fit, speed, scorecard, EN/DE vs local peers (own MLX port) | Gate failed — K8 (G2, G3, G4, G5); stopped and published | [`exp_036_kolibri_local_eval/`](./exp_036_kolibri_local_eval/) |
| Exp 037 | Kolibri through a forced gate: exp_036's questions re-registered behind a gate built from its bug hunt | Stopped — budget (plan STOP by rule); gate PASS | [`exp_037_kolibri_forced_gate/`](./exp_037_kolibri_forced_gate/) |
| Exp 038 | Kolibri as a drop-in answering model in CasaSol's COAPI glue | Complete — H1 refuted (ceiling-limited) | [`exp_038_kolibri_casasol_dropin/`](./exp_038_kolibri_casasol_dropin/) |
| Exp 039 | COAPI Voice on-policy: GRPO against a code-only verifier on Qwen3-4B (LoRA, attention only), same base and data as exp_035's SFT | Complete — no training gain (the Q4 +0.20 was quantisation); form not learned | [`exp_039_coapi_voice_onpolicy/`](./exp_039_coapi_voice_onpolicy/) |
| Exp 040 | COAPI Voice, correctness checkable: GRPO on Qwen3-4B against a reward that checks the final answer line (closed options, computed numbers, "not in the material"), offline dynamic sampling, imatrix Q4 | Complete — not robust (gain at Q4, none at Q8) | [`exp_040_coapi_voice_checkable/`](./exp_040_coapi_voice_checkable/) |
| Exp 041 | Answerability gate: a local decision model (Qwen3-4B prompted, Kev-4B, Clef-Flash 9B, Clef 27B on llama.cpp `/v1/systemone`) asks "can this material answer?" before the voice | Complete — no gate earns its place (over-blocking; all blocked answerable rows had lost their source in retrieval) | [`exp_041_answerability_gate/`](./exp_041_answerability_gate/) |
| Exp 042 | Answerability gate re-tested after fixing retrieval (Spanish via translation, two passages from the leading topic); answerable = source in the material | Complete — gate adopted (Clef 27B: abstention .474 → .921, 0/101 false refusals, exact .753 → .833; calibration fails) | [`exp_042_answerability_gate_retrieval/`](./exp_042_answerability_gate_retrieval/) |
| Exp 043 | Lead triage for an estate agency by one local decision model (Kev-4B), on 235 synthetic leads in 7 languages; code turns Kev's answers into the next action | Complete — usable for triage (intent .894, spam .906 caught / .015 false, next action .898; qualified and timeline fail; spam goes to review, never discarded) | [`exp_043_kev_lead_triage/`](./exp_043_kev_lead_triage/) |
| Exp 044 | Lead triage rev 4 against rev 3.1 on a fresh synthetic set (331 test leads, 7 languages, three hard strata): spam override only when Kev agrees, a "wants to own" question, budgets written as words | Complete — rev 4 adopted (false spam 5 → 2 of 282, budgets in words 0 → 29 of 29, request +0.013, next step unchanged; holiday-stratum gain +0.064 not significant; rev 3.1 replicates exp_043) | [`exp_044_kev_lead_triage_rev4/`](./exp_044_kev_lead_triage_rev4/) |
| Exp 045 | Lead triage from voice notes: exp_044's 331 test enquiries read aloud by macOS voices, encoded at WhatsApp voice-note quality, transcribed by Whisper large-v3-turbo on the mini, triaged by rev 4, paired against typed text; no cloud model | Complete — voice notes keep the triage (request 0.912 → 0.900, next step 0.931 → 0.921, false spam 2 → 3 of 282, budgets in words 0.966, language 1.00); stated budgets 0.763 → 0.701, 10 of 15 losses from the computer voice; reader gaps found | [`exp_045_voice_triage/`](./exp_045_voice_triage/) |
| Exp 046 | Translate before triage: gemma4:26b translates 425 non-English synthetic enquiries (exp_043 + exp_044 tests, six languages) into English for Kev, against Kev on the original; rev 4 frozen; no cloud model | Complete — no gain, Kev reads the original (request 0.927 native, 0.925 translated; next step 0.922, 0.920 (corrected 2026-10-20); Polish 0.853, 0.868); translation trades errors (fixes 4 valuation → sell, misses others) | [`exp_046_translate_first/`](./exp_046_translate_first/) |
| Exp 047 | A bigger decision model behind the same rules: Clef 27B answers rev 4.1's triage questions in Kev-4B's place on 566 synthetic enquiries (exp_043 + exp_044 tests, seven languages); questions, code facts, rules and cuts unchanged; no cloud model | Complete — the bigger model lifts the weak fields (qualified 0.821 → 0.936, timeline 0.712 → 0.833, request 0.919 → 0.991, next step 0.924 → 0.973, false spam 2 → 0); financing worse (0.923: already-approved mortgages read as no need); 13.5 s vs 1.7 s per enquiry, 29 GB | [`exp_047_clef_triage/`](./exp_047_clef_triage/) |
| Exp 048 | The middle-sized decision model: Clef-Flash 9B behind rev 4.1's rules on the same 566 enquiries, and a split where Kev-4B reads the request and Flash the facts; no cloud model | Complete — split the work (hybrid: request 0.919 = Kev, qualified 0.901, timeline 0.773, false spam 2); Flash alone reads the request better (0.965) but flagged 9 genuine clients writing through intermediaries as spam; adopted as rev 4.2 | [`exp_048_flash_hybrid/`](./exp_048_flash_hybrid/) |
| Exp 049 | Rev 5 against rev 4.2 on 350 fresh enquiries written from rev 4.2's recorded errors (Fable writers, Opus blind checker, gemma4:26b third reader): an on-behalf question, a price-only question, a dated timeline question, financing "use" | Complete — adopt on-behalf (intermediary false spam 2 → 0) and the dated timeline (bare-month timing 0.444 → 0.870, 23 vs 0); price-only off (+0.074, p 0.06); live as rev 5.1 | [`exp_049_kev_rev5/`](./exp_049_kev_rev5/) |
| Exp 050 | The front desk as a society of small models (Kev-4B + Clef-Flash 9B → rules → listing join → gemma4:26b writer → printed cards → audit) against gemma4:26b doing it all, on 247 fresh enquiries (Fable writers, Opus blind checker, invented inventory) | Complete — H1 safety fail (0.040 vs 0.065, p 0.16), H2 usefulness fail (0.758 vs 0.943, M thinking), H3 stability pass (1.000 vs 0.773), H4 4B writer pass, H5 privacy pass (0 contacts in 988 prompts) | [`exp_050_society_vs_one_model/`](./exp_050_society_vs_one_model/) |
| Exp 051 | MLX against Ollama on the same text, re-running Exp 011 (whose MLX runs read a repeated sentence): fixtures, Exp 011's fallback and natural enquiries, 99 clean calls | Complete — H1 pass (no MLX cliff to 40K), H2/H6 pass (MLX prefill 12–24 % faster), H3 fail (generation not faster), H4 fail (both runtimes faster than June), H5 pass (the repeated sentence did not change MLX speed) | [`exp_051_mlx_same_text/`](./exp_051_mlx_same_text/) |
| Watcher Run 001 | Adversarial watcher — CasaSol gap analysis | Complete | [`watcher_run_001/`](./watcher_run_001/) |
| Pre-Chronos benchmarks | Ad-hoc bench scripts (unnumbered) | Reference | [`experiments/`](./experiments/) |

For details on any entry, the `scientific_log.md` entry is authoritative.

## What this directory is not

This is not a code repository for shipping software. The scripts
under each experiment or incident are diagnostic instruments — they
exist to measure or reproduce something specific. They are
documented enough to re-run and to understand, but they are not
maintained as products.

This is also not a public engineering changelog. Posts on
localfirstai.eu are the public surface; this directory is the
substrate they reference. A post should be readable on its own and
the link to Chronos provides the receipts on demand.

## Hardware targets

| Machine | Chip | Memory | Role | Hostname |
| --- | --- | --- | --- | --- |
| Mac Mini (`miktam02`) | M4 Pro, 14-core CPU, 20-core GPU | 64 GB | Primary — always-on, cheap inference tier | `miktam-mini.local` |
| MacBook Pro 14" | M5 Max, 18-core CPU, 40-core GPU | 128 GB | Smart inference tier (home network only) | `miktam-mbp.local` |

All measurements unless otherwise stated are taken on the Mac Mini. Experiments specifying
MBP hardware are noted in the relevant log entry. Two-machine experiments: Exp 007, Exp 016.
