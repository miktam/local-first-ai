# exp_036: compute used

**Status:** final for exp_036 (gate failed; published 2026-10-05). Compiled 2026-10-06 on the mini.

**Labels:**
- **MEASURED** means the number comes from a run record, a timestamp or a transcript.
- **ESTIMATED** means no record covers it; the assumption is stated on the row.
- Energy is always ESTIMATED, because no run recorded watts on either machine.

**Machines:**
- **mbp:** MacBook Pro 14", M5 Max, 128 GB. It ran every real-weight Kolibri and peer-model run.
- **mini:** Mac mini, M4 Pro, 64 GB. It ran the build, the tests, the tiny-model and dry runs, the bug hunt and the small-model checks.

## Summary

| Quantity | Value | Label |
|---|---|---|
| mbp compute time | **3.8 h** (range 3.5–4.3 h): 3.07 h measured + 0.71 h estimated (0.43–1.21 h) | MEASURED + ESTIMATED |
| … of which real-weight GPU runs | 2.93 h | MEASURED |
| mbp downloads, 355 GB (not compute) | ≈ 4.6 h (3.5–6 h) | sizes MEASURED, times ESTIMATED |
| mini busy time (union of command intervals) | **8.87 h** (7.96 h without short commands) | MEASURED |
| Energy, both machines | **≈ 0.55 kWh** (0.33–0.89 kWh) | ESTIMATED |
| Claude tokens, mini session (claude-opus-5-5, 6,599 API responses) | uncached input 13,234 · cache write 42,725,343 (5 min 30,876,947, 1 h 11,848,396) · cache read 1,619,950,200 · output 2,009,544 | MEASURED |
| Claude tokens, the mbp's own session | **not measured** (transcripts not on the mini; see §5) | — |

**Not a speed measurement.** Every duration here is machine time, kept for accounting. None of them measures Kolibri's speed, and none should be read or converted as one. The gate failed, so the pre-registration's rule applies: no speed, memory or quality number from the ungated port is ever reported (HYPOTHESIS l.273). Mixed runs, such as the gate's reference passes, port checks and tests together, cannot be turned into tokens per second.

On the mbp, compute was 3.8 h out of a 47.0 h span, from 2026-10-03 15:58Z to the hand-back commit `2530ad4` at 2026-10-05 14:59Z. That is ≈ 8 % of the span.

The plan budgeted 35.1–39.9 h of mbp machine time. Almost all of it was for S1–S3: the bench, the pilot and the ≈ 30 h scored queue. None of S1–S3 ran, because the gate failed.

## 1. mbp machine time

**Sources:**
- the run records in this kit: `results/`, `aborted/`, `host/`, `diagnostics/gate1/out/` and `diagnostics/gate1/confirm/out/`;
- the HYPOTHESIS run records;
- the 29 mbp commits, which reached the mini by fast-forward pull; their times come from the mini's reflog.

**Measured runs** (UTC):

| Run | When | Duration |
|---|---|---|
| Phase 0 gate (`gate_20261005T050112Z`, the record's total) | 10-05 05:01:12–05:57:03 | 3,351 s |
| Gate-1 diagnostics, stages 0–4 (span of the stage headers) | 10-05 11:48:02–12:55:25 | 4,043 s |
| Peer checks 1, 2 and 3 | 10-04 and 10-05 | 722 s, 760 s, 1,066 s |
| Confirmation runs (U = 20261005T145122Z) | 10-05 14:51–14:58 | ≈ 428 s (to the minute) |
| Convert K8, then K4 | 10-04 13:33:32–13:34:55 | 51 s + 32 s |
| Step-3 unit tests, runs 1 and 2 | 10-04 | 207 s, 201 s |
| Preflight: quick, deep (aborted), deep | 10-04 | 1 s, 26 s, 26 s |
| Smoke test (2 generations) | 10-03 | ≤ 99 s (bound) |
| MLX `gather_qmm` reproductions on 0.31.2 and 0.32.3, plus the 0.32.3 install | 10-05 18:17 and 10-06 02:39–02:42 | 15.6 s, 14.5 s, 4.7 s |
| **Total** | | **11,048 s = 184.1 min** |

**Estimated, ≈ 42 min (25–73 min).** No record gives a duration for any of these:
- five more step-3 test runs, after Amendments 2–6, at ≈ 205 s each;
- the TF32 test rerun;
- the item-manifest builds and the gate texts;
- Gemma checks A–C (≈ 2 min) and D (≈ 4 min);
- the three preflights on 10-03 and the venv;
- leak and hash checks for the 29 commits (≈ 7 min);
- the `mlx_repro_min` and `repro.py` runs, which have no seconds field.

**Downloads, ≈ 4.6 h (ESTIMATED).**
- **Kolibri BF16, 156.2 GB:** ≈ 2.0 h, bounded at 1.93–2.73 h. That works out to ≈ 21.5 MB/s, the rate used for every other undated download.
- **Peer weights:** 147.2 GB.
- **Gemma bf16, for diagnostic D:** 51.6 GB.
- **Data:** 0.3 GB.
- **Machine active, compute and downloads together:** ≈ 8.3 h (ESTIMATED).

**Counted once:**
- The gate is counted by its total only. Its six phases add up to 3,349.3 s and are not added again. The 193 s of unit tests inside phase 1 are part of that total.
- The diagnostics are counted as one span. Stage 3's header covers only its 3.3 s report step. Its six compute steps ran as separate processes in the ≈ 38 min gap after stage 2, which lies inside the span.
- The confirmation runs are counted by their window. The measured parts (396 s) fall inside it.
- M4 Pro runs (`*_m4pro`, the bug hunt, stage 5) are mini work and are left out here.

## 2. mini machine time

Sources and method:
- **Calls timed:** every Bash and Monitor call in the mini's Claude Code session transcripts, both the main session and the 15 exp_036 workflows.
- **Duration:** each call runs from its `tool_use` to its `tool_result`. Background jobs run to their completion notice.
- **Excluded:** waits and polling, 34 calls and 1.06 h.

| Measure | Value |
|---|---|
| Calls counted | 6,045 |
| UNION, the wall time the mini was busy | **8.87 h** |
| SUM, which counts each parallel command separately | 11.04 h (average 1.23 commands running at once) |
| Busy time by day (UNION) | 10-03: 2.84 h · 10-04: 4.03 h · 10-05: 2.01 h |
| By category (UNION) | test suites 5.27 h · tiny-model, gate and dry runs, including the bug hunt 2.35 h · short commands 1.12 h · MLX inference on small and peer models 0.69 h · kernel probes 0.09 h · downloads, conversions and other 0.11 h |
| By energy class (GPU takes precedence over CPU) | GPU 3.05 h · CPU 5.83 h |

The category unions overlap, so they add up to more than 8.87 h.

What did not run on the mini:
- **No Ollama inference.** The only Ollama commands were the metadata calls `list`, `show` and `ps`.
- **No Kolibri weights.** The mini holds only Kolibri's config, index and tokenizer (14 MB).

## 3. Energy (ESTIMATED)

Energy is time × an assumed draw at the wall.

**mbp:**
- GPU runs: 60 / 90 / 130 W (low / central / high). The high case adds +10 % for battery charging.
- IO runs (tests, hashing, conversion): 25 / 35 / 55 W.
- Baseline: 0 W. The laptop would otherwise have been asleep, so all its run energy counts.

**mini:**
- The central draws come from spot readings with `macmon` (`sys_power`) on 2026-10-06, not taken during exp_036: 64.6 W under MLX GPU load, 5.8 W idle.
- Classes: GPU 40 / 65 / 85 W, CPU 10 / 18 / 35 W.
- Reported as **marginal** energy above the idle baseline of 5 / 6 / 10 W, because the mini is on around the clock anyway.

| Machine | Time used | Low | Central | High |
|---|---|---:|---:|---:|
| mbp | GPU 3.03 h, IO 0.75 h | 0.19 kWh | 0.30 kWh | 0.52 kWh |
| mini, marginal | GPU 3.05 h, CPU 5.83 h | 0.14 kWh | 0.25 kWh | 0.37 kWh |
| **Total** | | **0.33 kWh** | **0.55 kWh** | **0.89 kWh** |
| mbp awake while downloading (separate line, not in the total) | ≈ 4.6 h at 8 / 12 / 20 W | 0.03 kWh | 0.06 kWh | 0.12 kWh |

- **The low–high range** is a conservative bound, not a confidence interval.
- **Short commands** are charged as CPU time. That overstates the mini figure by at most 0.03 kWh.
- **Not converted:** network and hosting energy for the 355 GB of downloads, and Claude's cloud inference. No sourced per-GB or per-token figure exists for either.

## 4. Claude tokens (MEASURED)

All of this usage came from one Claude Code session on the mini, which started 2026-10-03 15:29:41Z, plus its agent workflows. The model was `claude-opus-5-5` throughout. The snapshot was taken 2026-10-06 12:39:02Z.

| Part | Messages | Uncached input | Cache write, 5 min | Cache write, 1 h | Cache read | Output |
|---|---:|---:|---:|---:|---:|---:|
| Main session | 533 | 1,102 | 0 | 11,848,396 | 243,458,510 | 746,150 |
| 15 workflows | 6,066 | 12,132 | 30,876,947 | 0 | 1,376,491,690 | 1,263,394 |
| **exp_036 total** | **6,599** | **13,234** | **30,876,947** | **11,848,396** | **1,619,950,200** | **2,009,544** |

How the tokens were counted and assigned:
- **De-duplication:** each API message id is counted once, from its final usage line. No id appears in more than one file.
- **Workflows** are assigned to exp_036 by name.
- **Main-session messages** are assigned by time window. The windows are set by Andrei's prompts.
- **The MLX `gather_qmm` follow-up** is counted here because it is committed to this record. It covers 48 messages and 71,542 output tokens.
- **Thinking tokens** are included in output.
- **Server-side web search and fetch:** 0 requests.

One compaction call, at 2026-10-05 14:46Z, left no usage record. It is ESTIMATED at ≈ 0.97 M input-side tokens and ≤ 19 k output (≈ 0.06 % of the input side). It is not added to the table.

No prices are applied. The cost depends on the plan: a subscription has no per-token charge, while the API charges each of the five token types at its own rate.

## 5. Not counted

- **The mbp's own Claude Code session.** Andrei handed it its prompt on 2026-10-03 at about 16:09Z, and it carried out the RUNBOOK steps and amendment syncs on the mbp. Its transcripts never reached the mini, so its tokens are neither measured nor estimated here. To measure them, run this on the mbp:
  `python3 mbp_tokens.py 2026-10-03T16:00:00Z 2026-10-06T02:48:00Z`
  It counts sessions whose cwd contains `local-first-ai`, de-duplicates by message id, and prints aggregate totals only.
- **Andrei's own time.**
- **Commands Andrei ran by hand on the mini**, and the mini's always-on services: the bots and the Ollama daemon.
- **mbp idle-awake time** over the 47 h span, apart from the download line in §3.
- **Agent thinking time.** It is not machine time; it is already in the tokens.
- **The accounting behind this record.** At 12:39Z it stood at ≥ 147 messages and 13,898 output tokens, and it is still growing. It is charged to neither experiment.
