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
| Exp 011 | MLX runtime vs Ollama context cliff | Complete | [`exp_011_mlx_runtime/`](./exp_011_mlx_runtime/) |
| Exp 012 | Cost vs capability curve | Complete | [`exp_012_cost_capability/`](./exp_012_cost_capability/) |
| Exp 013 | Local audit loop scaffolding | Complete | [`exp_013_local_audit_loop/`](./exp_013_local_audit_loop/) |
| Exp 014 | Capability variance floor | Complete | [`exp_014_capability_variance/`](./exp_014_capability_variance/) |
| Exp 015 | Active-parameter ablation (dense vs MoE) | Pre-registered | [`exp_015_active_param_ablation/`](./exp_015_active_param_ablation/) |
| Exp 016 | Two-mac orchestration | Phase B in progress | [`exp_016_two_mac_orchestration/`](./exp_016_two_mac_orchestration/) |
| Exp 017 | Argos Phase 0 — feed reconnaissance | Pre-registered | [`exp_017_argos_phase0/`](./exp_017_argos_phase0/) |
| Exp 018 | Sovereignty resilience (3 failure modes) | Pre-registered | [`exp_018_sovereignty_resilience/`](./exp_018_sovereignty_resilience/) |
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
