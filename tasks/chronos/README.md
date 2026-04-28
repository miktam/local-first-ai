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
├── exp_NNN_<slug>/            # one directory per experiment
├── incident_NNN_<slug>/       # one directory per investigated incident
└── experiments/               # ad-hoc benchmarks not promoted to a full experiment
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
| Experiment 003 | local memory anonymisation | landed | [`exp_003_local_memory/`](./exp_003_local_memory/) |
| Incident 003-Alpha | prefill scaling cliff | closed | [`incident_003_alpha/`](./incident_003_alpha/) |
| Latency benchmark v2 | ad-hoc | reference data | [`experiments/`](./experiments/) |

For details on any of these, the `scientific_log.md` entry is
authoritative.

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

## Hardware target

All measurements unless otherwise stated are taken on `miktam02`:
Apple M4 Pro Mac Mini, 14-core CPU (10P + 4E), 20-core GPU, 64 GB
unified memory. Specific model versions, runtime versions, and
configuration are recorded in the relevant log entry.
