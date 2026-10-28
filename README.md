# Local First AI

Benchmark data, experiment harnesses, and scientific logs for running production AI on local hardware — no cloud required.

This repo accompanies the blog at [localfirstai.eu](https://localfirstai.eu) and provides the verifiable evidence backing every claim made there.

**Contents:** [Hardware](#hardware) · [Experiments by theme](#experiments-by-theme) · [Key findings](#key-findings-cumulative) · [Blog posts](#blog-posts) · [Field notes](#field-notes) · [Build log](#build-log)

---

## Hardware

| Component | Value |
|---|---|
| Primary machine | Mac Mini M4 Pro (`miktam02`), 64 GB unified memory |
| Secondary machine | MacBook Pro M5 Max (`miktam-mbp`) — smart inference tier (Exp 016) |
| Primary model | `gemma4:26b` (MoE A4B, 25.8B total params / ~4B active per forward pass, Q4_K_M); the CasaSol bot's answering model, kept after MLX, QAT and Q8_0 were tested against it (Exp 052, 053) |
| Router model | `gemma4:e4b` (fast routing layer) |
| Decision models | Kev-4B, Clef-Flash 9B, Clef 27B: closed-question categorizers for gates and lead triage (Exp 041–050) |
| Runtime | Ollama 0.40.2 since 2026-10-10; 0.33.0 from 2026-08-29 (Exp 015); Exp 007 pre-registered 0.20.2 |
| MLX | mlx 0.32.4, mlx-lm 0.32.2 since 2026-10-24 (Exp 051) |
| Ollama settings | `OLLAMA_FLASH_ATTENTION=0`, default fp16 KV cache, `OLLAMA_CONTEXT_LENGTH=8192` (Exp 008/010, Exp 025) |
| Agent | Nestor (local AI agent); the OpenClaw gateway was retired on 2026-09-24 |
| Operating ceiling (Mini) | **> 40,000 tokens on-wire** (Exp 008/010 — no cliff at FA=0; Exp 007's 18K was a FA=1 artefact) |
| Operating ceiling (MBP)  | **> 40,000 tokens on-wire** (Exp 007 Phase B — cliff at ~45K; the MacBook's Ollama flags were not recorded) |

---

## Field notes

Some posts describe operational work rather than experiments: what we observed, what we changed, and the evidence. Each has a field note with the data behind every claim and a script to re-check what can still be checked. Field notes are recorded after the fact, so the Chronos rule (hypothesis registered before the run) does not apply to them, and their claims are limited to what was observed.

| Date | Field note | Post |
|---|---|---|
| 2026-10-18 | [Website hygiene: aprimerose.com](./field-notes/2026-10-18-website-hygiene/) | 80% of Our Web Traffic Was a Dead Subdomain Redirecting to Itself |

---

## Build log

A dated line for each change to the local tools: what changed, what prompted it (a real input, one of our tests, or reading), how it was checked, and what was learned. Model notes keep what we know about each model, one sourced fact per line.

| | |
|---|---|
| Log | [log/](./log/) · [October 2026](./log/2026-10.md) |
| Model notes | [Kev-4B](./log/models/kev-4b.md) · [Jev](./log/models/jev.md) |

---

## Project Chronos

Every experimental claim on the blog is backed by a pre-registered experiment logged in [`tasks/chronos/scientific_log.md`](./tasks/chronos/scientific_log.md). The methodology: observation → hypothesis → experiment → evidence → conclusion. No retrofitted results.

Roadmap and pending experiments: [`tasks/chronos/roadmap.md`](./tasks/chronos/roadmap.md)

### Experiments by theme

One line per experiment. The registration, the full result and the evidence are in each experiment's folder; the log holds the running record ([`scientific_log.md`](./tasks/chronos/scientific_log.md)), and the [Chronos index](./tasks/chronos/README.md) lists every experiment in order.


#### Runtime and hardware on Apple Silicon

| # | Experiment | Result |
|---|---|---|
| [002](./tasks/chronos/exp_002_control_plane_vs_data_plane/) | Control plane vs data plane | Thinking runs flat at ~38 t/s until an unconstrained prompt runs away: cap it and name it |
| [007](./tasks/chronos/exp_007_hardware_comparison/) | Mac mini vs MacBook Pro M5 Max | The M5 Max is faster, but the Mini's 18K cliff came from a flag (see 008); the MacBook's flags were not recorded |
| [008](./tasks/chronos/exp_008_flash_attention/) | Flash attention + q8_0 KV cache | The flags caused the cliff: with both off, no cliff through 40K |
| [010](./tasks/chronos/exp_010_fa_isolation/) | Flash attention vs q8_0, factorial | Flash attention alone causes the cliff; q8_0 alone is benign |
| [011](./tasks/chronos/exp_011_mlx_runtime/) | MLX runtime vs Ollama | No MLX cliff, but its runs read a repeated sentence instead of the fixtures; re-run in 051 |
| [016](./tasks/chronos/exp_016_two_mac_orchestration/) | Two-Mac orchestration | In progress: the LAN path from the mini to the MacBook works; timing pending |
| [023](./tasks/chronos/exp_023_local_model_efficiency/) | Generation efficiency across local models | gemma4:26b has the best wall-clock of four; qwen3.8:27b finds more but takes 6.7× longer |
| [025](./tasks/chronos/exp_025_context_config_isolation/) | Context-allocation isolation | An unset `OLLAMA_CONTEXT_LENGTH` made gemma4:31b ~100× slower; one config line fixed it |
| [051](./tasks/chronos/exp_051_mlx_same_text/) | MLX vs Ollama on the same text | MLX reads prompts 12–24 % faster and generates no faster; neither has a cliff |

#### Security, privacy and sovereignty

| # | Experiment | Result |
|---|---|---|
| [003](./tasks/chronos/exp_003_local_memory/) | Anonymized adversarial memory | 0/20 source recognition, 0/3 identity leaks: the boundary is architectural |
| [006](./tasks/chronos/exp_006_redactor_fidelity/) | Redactor fidelity (GDPR) | No leak across 20 toxic notes and 8 GDPR categories |
| [018](./tasks/chronos/exp_018_sovereignty_resilience/) | Sovereignty resilience | Daemon down and weights removed both degrade gracefully; the network cut also blinded the observer |
| [019](./tasks/chronos/exp_019_adversarial_legal_review/) | Adversarial legal review | 0 of 7 claims survived three review rounds unchanged |
| [020](./tasks/chronos/exp_020_miktam_mini_hardening/) | Hardening the inference node | Unauthenticated Ollama fixed; an unrestricted `tail` grant was a real root-read path |
| [021](./tasks/chronos/exp_021_independent_red_team/) | Independent red-team pass | Credentials readable without any escalation; rotation declined as a documented risk |
| [022](./tasks/chronos/exp_022_bot_red_team/) | Red-team of the CasaSol bot | Injection, extraction and bypass refuted; corpus poisoning through `/witness` confirmed |
| [054](./tasks/chronos/exp_054_exposure_review_egress/) | Exposure Review: what leaves the machine | 4 of 5 written claims held; platform link shorteners were looked up. Fixed in Parry: all 5 hold on re-run |
| [055](./tasks/chronos/exp_055_exposure_review_lures/) | Exposure Review: targeted-lure benchmark | The check catches 75 % (all new-domain lures, 1 of 14 link-free); gemma alone 84 %, Opus 95 % |
| [056](./tasks/chronos/exp_056_triage_injection/) | Exposure Review: can a message subvert its own check | Registered and parked before any run; payloads frozen by hash |
| [057](./tasks/chronos/exp_057_local_inference_egress/) | Does local inference leave the machine (Ollama runtime, nine Qwen/gemma subjects, whole machine watched) | Registered 2026-10-28; not yet run |

#### Local models against frontier models on analysis

| # | Experiment | Result |
|---|---|---|
| [009](./tasks/chronos/exp_009_adversarial_critic/) | Adversarial project critic | Fail: the local critic missed the implementation gap; shared compliance findings were partly false |
| [012](./tasks/chronos/exp_012_cost_capability/) | Cost vs capability | gemma4:26b 0/8, frontier 5/8; Haiku's 5/8 did not replicate (014) |
| [013](./tasks/chronos/exp_013_local_audit_loop/) | Local audit loop | Decomposition recovered 2 of 3 findable items for gemma4:26b |
| [014](./tasks/chronos/exp_014_capability_variance/) | Capability variance floor | gemma4:26b never above 0/8; Haiku 1–4/8 over five runs |
| [015](./tasks/chronos/exp_015_active_param_ablation/) | Active-parameter ablation | Registered: dense 12–32B against MoE A4B on the audit rubric |

#### Property-law answers: training against glue

| # | Experiment | Result |
|---|---|---|
| [026](./tasks/chronos/exp_026_contextual_retrieval/) | Contextual retrieval, fully local | Prefixed chunks raise exact hits from 0.455 to 0.636 (pilot, n = 11) |
| [035](./tasks/chronos/exp_035_coapi_voice/) | Form in the weights, facts in retrieval | Three LoRA runs cost substance every time; the untrained 4B behind glue wins |
| [039](./tasks/chronos/exp_039_coapi_voice_onpolicy/) | GRPO against a code-only reward | No training gain at Q8: the Q4 gain was our quantiser |
| [040](./tasks/chronos/exp_040_coapi_voice_checkable/) | GRPO with a checkable reward | +6.7 points at Q4, +1.3 (not significant) at Q8: not robust |
| [041](./tasks/chronos/exp_041_answerability_gate/) | Answerability gate | No gate earned its place: its false refusals were retrieval misses |
| [042](./tasks/chronos/exp_042_answerability_gate_retrieval/) | Answerability gate, retrieval fixed | Adopted: Clef 27B refusals 0.47 → 0.92, none of 101 answerable refused |

#### New open models

| # | Experiment | Result |
|---|---|---|
| [024](./tasks/chronos/exp_024_vision_model_comparison/) | Vision: gemma4:26b vs qwen3.8:27b | Inconclusive: 1/5 valid JSON; qwen vanished from disk mid-run |
| [036](./tasks/chronos/exp_036_kolibri_local_eval/) | Kolibri on a MacBook (our MLX port) | Our own gate failed the port; no scores published |
| [037](./tasks/chronos/exp_037_kolibri_forced_gate/) | Kolibri through a forced gate | Gate passed; stopped on budget (56 h against 31 h) |
| [038](./tasks/chronos/exp_038_kolibri_casasol_dropin/) | Kolibri as a drop-in answering model | −0.10 against gemma4:26b; no German edge |

#### Lead triage and the front desk

| # | Experiment | Result |
|---|---|---|
| [043](./tasks/chronos/exp_043_kev_lead_triage/) | Lead triage with one decision model | Usable as a suggestion: request 0.89, next step 0.90; 3 of 203 genuine flagged as spam |
| [044](./tasks/chronos/exp_044_kev_lead_triage_rev4/) | Triage rev 4 | Adopted: false spam 5 → 2, budgets written in words 0 → 29/29 |
| [045](./tasks/chronos/exp_045_voice_triage/) | Triage from voice notes | Voice notes keep the triage; budgets are the weak spot |
| [046](./tasks/chronos/exp_046_translate_first/) | Translate before triage | No gain: the decision model reads the original |
| [047](./tasks/chronos/exp_047_clef_triage/) | A bigger decision model | Clef 27B lifts the weak fields (request 0.99) at 13.5 s and 29 GB |
| [048](./tasks/chronos/exp_048_flash_hybrid/) | The middle model, and a split | Adopted: Kev reads the request, Flash the facts |
| [049](./tasks/chronos/exp_049_kev_rev5/) | Rev 5: targeted fixes | Telling the model the date fixed month-only timing (0.44 → 0.87) |
| [050](./tasks/chronos/exp_050_society_vs_one_model/) | A team of small models vs one model that thinks | Not safer, less useful (0.76 against 0.94), but the only stable one |

#### The CasaSol bot's answering model

| # | Experiment | Result |
|---|---|---|
| [052](./tasks/chronos/exp_052_bot_on_mlx/) | The bot on MLX | 10 % slower; blind judges preferred its answers 39 to 22; stays on Ollama |
| [053](./tasks/chronos/exp_053_precision_behind_the_bot/) | Higher-precision builds behind the bot | Q8_0 preferred but wrong more often and 17 % slower; QAT worse; keep Q4_K_M |

#### Framework and early work

| # | Experiment | Result |
|---|---|---|
| [001](./tasks/chronos/exp_001_verification_of_veracity/) | Verification of veracity | The Chronos framework, activated |
| [004](./tasks/chronos/exp_004_bootstrap_diet/) | Bootstrap diet | Agent session hygiene |
| [005](./tasks/chronos/exp_005_dicer_describer/) | Router / Reducer cascade | A two-model cascade over an 8-year Apple Watch corpus |
| [017](./tasks/chronos/exp_017_argos_phase0/) | Argos phase 0 | Registered: EV-corridor feed reconnaissance |

### Watcher Runs

Production runs of the Adversarial Watcher — a staged local LLM pipeline that compares documented intent against shipped artefacts and produces annotated gap reports. Each run is an auditable Chronos artefact.

| Run | Project | Confirmed gaps | False positives | Evidence |
|---|---|---|---|---|
| [watcher_run_001](./tasks/chronos/watcher_run_001/) | CasaSol | 5 | 3 | 2026-06-06, gemma4:26b, ~270s |

### Incidents

| # | Name | Finding |
|---|---|---|
| [003-Alpha](./tasks/chronos/incident_003_alpha/) | Memory Bandwidth Cliff | Prefill on `gemma4-think:26b` goes super-quadratic past ~25K tokens on Apple Silicon. Hard operational ceiling: **< 22K tokens on-wire**. The bottleneck is memory bandwidth, not VRAM. ⚠ Superseded: Exp 008/010 traced the cliff to `OLLAMA_FLASH_ATTENTION=1`; with it off there is no cliff through 40K. |

---

## Benchmarks

Early benchmarks that preceded the Chronos framework. Scripts and results in [`tasks/chronos/experiments/`](./tasks/chronos/experiments/).

| Script | What it measures |
|---|---|
| `nestor-bench-phase1.sh` | Context window (4K–130K) vs generation speed. Finding: gen_tps flat at ~41 t/s. |
| `nestor-bench-phase1b.sh` | Thinking mode token overhead. Finding: 5–15× token cost for zero quality gain on simple tasks. |
| `nestor-bench-phase2-compare.sh` | Compressed-memory retrieval vs raw context. |
| `nestor-bench-phase2-memory.sh` | Memory layer latency at scale. |
| `nestor-bench-phase2b-retrieval.sh` | Retrieval accuracy across compression levels. |

```bash
ollama pull gemma4:26b
chmod +x tasks/chronos/experiments/nestor-bench-phase1.sh
./tasks/chronos/experiments/nestor-bench-phase1.sh
# Results written to tasks/chronos/experiments/results/
```

---

## Key findings (cumulative)

Numbered in the order they were found; grouped by theme.


### Runtime and hardware

- **(1)** **The prefill cliff was an artefact of `OLLAMA_FLASH_ATTENTION=1`, not a hardware limit.** Exp 007 measured a cliff at ~18K tokens on Mac Mini M4 Pro — but those runs were made with FA=1+q8_0 enabled (Incident 007-Alpha). Exp 008 established the FA=0 baseline: no cliff through 40K tokens. Exp 010's 2×2 factorial confirmed FA=1 is the sole culprit: it alone (without q8_0) produces a cliff at 32.5K and triples prefill latency at 15K tokens (1.774 → 5.405 ms/tok). With FA=0 (with or without q8_0), the Mac Mini's operational ceiling is **> 40K tokens on-wire**. Our explanation, not measured: Flash Attention was designed for discrete GPU SRAM/HBM hierarchies, and on Apple Silicon unified memory its tiling overhead applies without the bandwidth benefit. (Incident 003-Alpha, Exp 007, Incident 007-Alpha, Exp 008, Exp 010)

- **(5)** **Superseded: the 22K ceiling was not a property of the hardware.** We first read it as memory bandwidth saturating during prefill (Incident 003-Alpha), and built mitigations for it: cliff-aware coarsening in the extractor, hard token budgets in the cascade, a streaming watchdog. Exp 008 and Exp 010 traced the cliff to `OLLAMA_FLASH_ATTENTION=1` (finding 1).

- **(6)** **The M5 Max is faster, but Exp 007 overstated the gap.** At 25K tokens the MBP generated 66 t/s against the Mini's 14, but the Mini ran with flash attention and q8_0 on (Incident 007-Alpha). With both off the Mini generates 31 t/s (Exp 008), so the gap is about 2×. MBP prefill at 35K (1.24 ms/tok) is still below the flags-off Mini's at 4K (1.56 ms/tok). The MBP's own Ollama settings were not recorded. (Exp 007, Exp 008; corrected 24 October 2026)

- **(8)** **MLX reads long prompts faster than Ollama on this machine, and generates no faster.** Exp 011's MLX script missed its fixture files and measured a repeated sentence, so its comparison with Ollama was on different text and lengths. Exp 051 re-ran it on the same text, the same day: no prefill cliff on MLX to 40K, prefill 12–24 % faster than Ollama on the fixtures and 15–22 % faster on natural text, and generation level at 4K but 4–22 % slower from 8K. For an 8K prompt with a 300-token answer that saves about 1.2 s, with different weights (5.0 bits per weight against Q4_K_M), so a move to MLX needs its own quality test. No experiment here measured memory bandwidth. (Exp 011, Exp 051)

- **(2)** **Thinking tokens are expensive — cap and name them explicitly.** `gemma4:26b` is a thinking model: left unconstrained, a simple task generates 10,000–25,000 hidden thinking tokens — at 38 t/s that's 4–11 minutes per response with zero quality gain. The architectural response: a `gemma4-think:26b` Ollama alias with a hard 128K context cap, used only for tasks that genuinely need deliberation. The name makes the choice visible; the cap prevents runaway. (Exp 002)

- **(4)** **A two-model cascade extends the operating envelope.** Router (`gemma4:e4b`) routes in ~3–4s. Reducer (`gemma4:26b`) synthesises only what fits below the 22K cliff. The cascade made an 8-year health corpus queryable on local hardware without hitting the bandwidth cliff on normal queries. (Exp 005)


### Security, privacy and sovereignty

- **(3)** **Data sovereignty is an architectural property, not a policy.** An anonymization boundary enforced by the import graph — not by a prompt or a config flag — defeated source recognition (0/20) and identity bridging (0/3) on a corpus the model has memorised. The moat is the architecture. (Exp 003)

- **(7)** **A fixed redaction prompt reliably produces GDPR-clean output.** 20 synthetic toxic real estate notes spanning 8 pre-registered GDPR categories — 0 true-positive leaks in any output. The local 26B model with `temperature=0.1` and a structured system prompt passes all four pre-registered criteria: zero leaks, full structural compliance (TAGS + DESCRIPTION), all 20 within 300s. (Exp 006)

- **(16)** **Exposure Review's message check is a precise domain-age detector, not yet a lure detector.** In a benchmark of 110 targeted lures and 110 genuine messages built from a fictional client's public footprint, Parry's rules never flagged a genuine message but caught 62 % of lures; with its local model step, 75 % at 13 % false alarms. It caught every lure on a new domain, 10 of 24 on aged domains and 1 of 14 with no link at all. gemma4:26b alone, given the whole footprint, caught 84 % at 7 %; Opus 95 % at 5 %. What leaves the machine during a review is as documented; the one gap, platform link shorteners being looked up, is fixed and re-tested. (Exp 054, 055)


### Local models against frontier models

- **(9)** **Only the frontier critic caught the implementation gap, and the shared compliance findings were partly false.** A head-to-head adversarial critic comparison (three fixed personas, fixed JSON schema) found gemma4:26b and Claude Sonnet 4.6 agreeing on three compliance findings (DPA template, DSAR procedure, DPIA). Two of them were false: the DPA and DSAR documents existed (corrected 24 October 2026). gemma4:26b missed the highest-severity engineering finding: a primary moat component described across the BRIEF, deck, and BUILD_LOG had no corresponding code in any commit. gemma4 pattern-matched on documented claims and critiqued their replicability; Claude cross-referenced the BUILD_LOG claim against the git history and flagged the absence. Overlap rate: ~50%. False-positive rate: ~50%. Verdict: FAIL as a drop-in replacement. Exp 012 and Exp 014 later scored gemma4:26b (thinking off) 0 on a compliance rubric. (Exp 009, Exp 012, Exp 014)

- **(10)** **The cost-capability curve has one step — confirmed at the 4B-active/frontier boundary; dense local models untested.** ⚠ *Scope correction (Exp 012-Alpha, 2026-06-09): gemma4:26b is MoE A4B (~4B active parameters per forward pass, not 25.8B). Exp 012 tested one local model class; whether the step holds for dense 12–32B local models is open and pre-registered as Exp 015.* A four-model sweep (gemma4:26b, Haiku, Sonnet, Opus) on a pre-scored 8-point rubric across two task types (DPO compliance extraction + engineering gap detection) produced: gemma4:26b 0/8, all three frontier models 5/8 net. Haiku→Sonnet (3.1× cost) and Haiku→Opus (6.4× cost) each yield zero additional rubric points. For structured analytical extraction on bounded context (~10K tokens), Haiku is cost-dominant. Qualitative differences exist within the same net score: Haiku and Sonnet have complementary blind spots (Haiku misses concurrency risk; Sonnet misses auth gap). Opus has the highest gross score (6/8) but the highest false positive rate. Two items — a missing DPIA for VLM processing and the Art. 22 model-version accountability gap — evaded all four models. ⚠ Haiku's 5/8 did not replicate: Exp 014 ran it five more times the same evening (1, 1, 4, 1 and 2 of 8); Sonnet and Opus were not re-run. The costs cover three runs of both tasks; one audit costs about a third. (Exp 012, Exp 014; scope corrected 2026-06-09 — see Exp 012-Alpha in scientific_log.md; corrected 24 October 2026)


### Small models, glue and training

- **(11)** **At a few hundred examples, training a small model cost more than it bought; deterministic glue did not.** Three LoRA runs and two GRPO runs on Qwen3 1.7B and 4B left correctness no better than the untrained base, robustly. Each LoRA run lowered the model's use of retrieved facts. The reference build is the untrained 4B behind retrieval and code. (Exp 035, Exp 039, Exp 040)

- **(12)** **A local decision model in front of a small voice decides when not to answer.** With retrieval fixed first, Clef 27B as a gate raised correct refusals from 0.474 to 0.921 and wrongly refused none of 101 answerable questions; Kev-4B came close at a sixth of the time. (Exp 041, Exp 042)

- **(13)** **For lead triage, a decision model answers closed questions and the agency's rules decide.** Kev-4B with rules in code read the request right about 90 % of the time on synthetic enquiries; a bigger decision model behind the same rules lifted the weak fields, so the 4B, not the rules, was the ceiling. Translating first gained nothing; telling the model today's date lifted timing given only as a month from 0.44 to 0.87. (Exp 043–049)

- **(14)** **A team of small models was more stable than one model that thinks, not safer, and less useful.** On 247 synthetic enquiries, the team sent a wrong reply 4.0 % of the time against 6.5 % for gemma4:26b thinking alone (p 0.16), sent 0.758 of the good replies against 0.943, made the same decision in all three runs (one model changed 56 of 247), and ran 6× faster. (Exp 050)

- **(15)** **For the live bot, what limits answer quality is content, not the runtime or the weights.** On the bot's own workload (short prompts, about 1,200 thinking tokens per answer), MLX was 10 % slower than Ollama, QAT 10 % faster but worse, and Q8_0 better liked but wrong more often and 17 % slower; Q4_K_M on Ollama stays. The errors every build shared were in what it was given: a contributor's name, course example figures read as market data, and legal facts the material had wrong or did not hold. Those are now fixed in the bot; their effect has been checked on six questions, not measured. (Exp 051, 052, 053)

---

## Blog posts

Published at [localfirstai.eu](https://localfirstai.eu):

**Technical — benchmarks, experiments, architecture**

- [Our Team of Small Models Against One Model That Thinks: More Stable, Less Useful](https://localfirstai.eu/posts/2026-10-24-one-model-that-thinks/) — Exp 050: an estate agency's front desk as a team of small models and code, against gemma4:26b thinking alone. Not safer (wrong replies 4.0 % against 6.5 %, p 0.16), less useful (0.758 of good replies sent against 0.943), the same decision every run, 6× faster.
- [We Blamed Our Rules. The 4B Model Was the Ceiling.](https://localfirstai.eu/posts/2026-10-21-the-model-was-the-ceiling/) — Exp 045–049: a 27B decision model behind exactly the same rules read the request right 0.991 of the time against 0.919. What shipped is two small models splitting the work; telling the model today's date nearly doubled its accuracy on timing given only as a month or holiday.
- [A Second AI Doubted 1 in 5 Test Enquiries. We Deleted Those.](https://localfirstai.eu/posts/2026-10-20-synthetic-data-lessons/) — Seven things synthetic data did to our experiments since September, each with its record. One AI wrote the test enquiries, a second labelled them again, and we deleted the ones the second doubted: nearly one in five. Counts corrected 24 October 2026.
- [Sorting Estate-Agency Enquiries on a Mac Mini: 89 % Right on 235 Synthetic Messages](https://localfirstai.eu/posts/2026-10-19-lead-triage/) — Exp 043: one local decision model answers closed questions; the agency's rules in code pick the urgency and the next step. Spam goes to a review folder, never deleted. Baselines added corrected 24 October 2026.
- [A Local Decision Model Made Our 4B Refuse 92 % of Unanswerable Questions and None of the Answerable Ones](https://localfirstai.eu/posts/2026-10-18-answerability-gate/) — Exp 041/042: no gate earned its place until retrieval was fixed; then Clef 27B raised refusals from 0.474 to 0.921 with no false refusals.
- [80% of Our Web Traffic Was a Dead Subdomain Redirecting to Itself](https://localfirstai.eu/posts/2026-10-18-dead-subdomain-redirecting-to-itself/) — Field note: 230,680 requests in a week, most of them one dead subdomain redirecting to itself. Fixing it turned up six quieter problems, email included.
- [Our 4B Property Model Gained 6.7 Points at Q4 and None at Q8](https://localfirstai.eu/posts/2026-10-17-checkable-reward-gain-vanished-at-q8/) — Exp 040: a reward that reads the final answer line. The gain at Q4 vanished at Q8; the quantiser flips about as many answers as training did.
- [We Trained Our 4B Property Model With a Code-Checked Reward. It Did Not Get More Correct.](https://localfirstai.eu/posts/2026-10-15-code-checked-reward-did-not-make-it-more-correct/) — Exp 039: GRPO against a reward computed only by code. Our audit stopped run 1 for gaming; run 2 seemed to gain +0.20 until a Q8_0 control showed the gain was our quantiser's damage to the base. The model learned to cite what it retrieved; it did not get more correct. Why, and what we'll change. Training-trace reading corrected 24 October 2026.
- [Kolibri Scored Below Gemma 4 in Our Property Assistant](https://localfirstai.eu/posts/2026-10-10-kolibri-scored-below-gemma-4/) — Exp 038: Aleph Alpha's Kolibri-1, untrained at 4-bit, swapped in for gemma4:26b behind CasaSol's answering pipeline. −0.10 on 60 sealed questions, no German edge, native German on 13 of 36 answers against gemma's 30. No live-bot trial.
- [The Part That Looked Fine](https://localfirstai.eu/posts/2026-09-30-the-part-that-looked-fine/) — The Malaga-AI study group's 2026 cohort: four projects on testing AI (sycophancy, bias in funding narratives, political flattery across 51,000 calls, self-rewriting instructions). In three, the first score was reassuring and a second one wasn't.
- [We Trained It Three Times. Then We Stopped.](https://localfirstai.eu/posts/2026-09-22-we-trained-it-three-times-then-stopped/) — Exp 035: three LoRA runs installed a professional's form in a phone-sized model and lowered its use of retrieved facts every time. The build that won has no trained weights: untrained Qwen3-4B behind retrieval and deterministic post-processing. Corrected 24 October 2026: the glue met 5 of 7 format floors, not 7.
- [We Stopped the Kolibri Benchmark Before Scoring One Answer](https://localfirstai.eu/posts/2026-10-09-we-stopped-the-kolibri-benchmark/) — Exp 037: our MLX port of Kolibri passed a rebuilt, forced-routing gate on runs 2 and 3, after two fixes outside the port. Kolibri's measured answer lengths then put the smallest scored run at 56 h against a 31 h budget, and the frozen plan rule stopped it. Four bench numbers sit beside their thresholds with no verdicts. Also: an mlx-lm 0.32.0 buffer leak that crashes long batched runs, and the workaround.
- [We Ported Aleph Alpha's Kolibri to a MacBook. Our Own Gate Said No.](https://localfirstai.eu/posts/2026-10-05-our-own-gate-said-no/) — Exp 036: our own MLX port of Kolibri failed the blocking gate we pre-registered, so the post has no scores. Run twice with only its chunk size changed, the port picks different experts and drifts apart at long context. On our M5 Max, a probe of mlx-lm's batched expert layer returned wrong numbers.
- [We Found the Credentials. We Didn't Rotate Them.](https://localfirstai.eu/posts/2026-09-20-we-found-the-credentials/) — Exp 021: a zero-escalation red-team pass found live credentials readable with one `cat`. We chose not to rotate them, wrote down exactly why, and published the risk register — including the part where the audit tooling leaked the secrets itself.
- [The Model Wasn't Broken](https://localfirstai.eu/posts/2026-08-30-the-model-wasnt-broken/) — Exp 023/025: a model that looked dead was 400× slower because `OLLAMA_CONTEXT_LENGTH` was never set and Ollama auto-selected 256k context for every call. One config line fixed it, zero code changes; gemma4:26b still wins on wall-clock against the newest Qwen.
- [We Tested What We Didn't Notice](https://localfirstai.eu/posts/2026-08-22-we-tested-what-we-didnt-notice/) — Exp 018: daemon-down and weights-removed both degraded gracefully with zero data loss. The network-cut test severed the only channel to the machine and forced a reboot — sovereignty of inference is not sovereignty of observability.
- [We Red-Teamed Our Own Bot](https://localfirstai.eu/posts/2026-07-22-we-red-teamed-our-own-bot/) — Exp 022: prompt injection, session extraction, and guardrail bypass all refuted. Corpus poisoning via the `/witness` community channel confirmed — one admin approval turns a fabricated claim into fact for every subsequent user.
- [Hardening the Inference Node](https://localfirstai.eu/posts/2026-07-08-hardening-the-inference-node/) — Exp 020: auditing the machine behind the local-first pitch. A static permission listing's worst-looking finding wasn't real; the one that was real (`tail`, NOPASSWD) wasn't the one that looked scary on paper.
- [We Reviewed Our Own Legal Brief with an Adversarial AI Panel. Zero of Seven Claims Survived Unchanged.](https://localfirstai.eu/posts/2026-06-24-adversarial-legal-panel/) — Exp 019: Claude adversarial panel + GPT-4o citation check + Gemini barrister review. 0/7 claims survived. BDAA omission, ingestion transfer gap, Gibraltar-EEA adequacy gap.
- [We Didn't Notice](https://localfirstai.eu/posts/2026-06-13-we-didnt-notice/) — The US government suspended the world's best AI model overnight for all foreign nationals. CasaSol was unaffected. Exp 018 pre-registered.
- [The Cost-Capability Curve Has One Step](https://localfirstai.eu/posts/2026-06-09-cost-capability-curve/) — A four-model sweep at the 4B-active/frontier boundary. One step, not a ramp. Haiku is cost-dominant; Haiku→Opus is 6.4× cost, 0 score gain. Corrected 24 October 2026: Haiku's 5/8 did not replicate, and the costs were for six calls, not one audit.
- [Same Hardware. Different Runtime. Same Result.](https://localfirstai.eu/posts/2026-06-09-mlx-vs-ollama-runtime/) — Exp 011: MLX and Ollama FA=0 on the same Mac Mini M4 Pro. Corrected 24 October 2026: the MLX runs used one repeated sentence instead of the fixtures, so the two runtimes saw different prompts. Exp 051 re-ran it on the same text: MLX reads prompts 12–24 % faster and generates no faster.
- [The Cliff That Wasn't](https://localfirstai.eu/posts/2026-06-07-the-cliff-that-wasnt/) — The 20K prefill cliff that shaped six months of cascade architecture was `OLLAMA_FLASH_ATTENTION=1`. Removing it tripled the Mac Mini's operational ceiling to >40K tokens. Full 2×2 factorial: FA=1 is the sole culprit, q8_0 alone is benign. Table values corrected 24 October 2026.
- [The Adversarial Watcher: When a Local Model Audits Its Own Project](https://localfirstai.eu/posts/2026-06-06-adversarial-watcher/) — A staged 5-step pipeline that catches documentation drift before every merge. First production run: 5 confirmed gaps, 3 false positives, anatomy of each.
- [We Tried to Replace Claude with a Local Critic. Here's Exactly Where It Failed.](https://localfirstai.eu/posts/2026-06-06-adversarial-critic/) — Exp 009: head-to-head adversarial review. Only the frontier model caught the impl-vs-docs gap. Corrected 24 October 2026: two shared compliance findings were false, and gemma4:26b later scored 0 on a compliance rubric.
- [The Silicon Wager: M4 Pro vs M5 Max](https://localfirstai.eu/posts/2026-05-29-silicon-wager/) — Exp 007: every Chronos envelope was measured on one machine. A second arrived. Corrected 24 October 2026: the Mini's wall came from Ollama settings, not the chip.
- [The GDPR Canary for Real Estate: 8 Data Categories, 0 Leaks](https://localfirstai.eu/posts/2026-05-09-redactor-fidelity/) — Exp 006: pre-registered fidelity sweep over 20 synthetic toxic notes. Zero true leaks. The claim becomes evidence.
- [The Memory Bandwidth Cliff](https://localfirstai.eu/posts/2026-04-28-incident_003_alpha_post/) — Incident 003-Alpha: why local AI is bound by the bus, not the GPU. Superseded by The Cliff That Wasn't: the cliff was a flag.
- [The Architecture of Anonymity](https://localfirstai.eu/posts/2026-04-26-the-architecture-of-anonymity/) — Exp 003: data sovereignty enforced by the import graph, not by policy.
- [The Control Plane and the Data Plane](https://localfirstai.eu/posts/2026-04-22-control-plane-vs-data-plane/) — Exp 002: managing the AI thinking tax.
- [The Genesis of Chronos](https://localfirstai.eu/posts/2026-04-21-genesis-of-chronos/) — Why Nestor commits to verified, evidence-backed claims.

**Essays — strategy, product, philosophy**

- [The Unfakeable Layer](https://localfirstai.eu/posts/the-unfakeable-layer/) — As generation gets cheap and agent-mediated, value moves to what a model can't fake: proof of work, human judgment, original ideas. Notes from Startup Olé Marbella.
- [Why CasaSol.ai](https://localfirstai.eu/posts/2026-05-22-why-casasol/) — If every company can be a Palantir now, how do you test that claim? An attempt to answer by building one — on the Costa del Sol.
- [Should We Stop Asking Local LLMs to Think?](https://localfirstai.eu/posts/should-we-stop-asking-local-llms-to-think/) — What Adam Smith, neuroscience, and a melting Mac Mini taught me about the real division of cognitive labour.
- [The Sovereign Individual: Why Private Data is the Only Moat Left](https://localfirstai.eu/posts/the-sovereign-individual/) — As AI becomes commoditised, competitive advantage is private context.
- [Every Company Can Be a Palantir Now](https://localfirstai.eu/posts/every-company-can-be-a-palantir-now/) — Proprietary structured data is the durable moat.

---

## License

MIT
