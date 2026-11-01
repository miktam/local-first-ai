# Does this record publish its failures? A check against the literature's bias

*2026-11-01. Prompted by reading, not by an incident: The Economist's four-part "AI eats science" package (Oct 12–28 2026) reports that scientific publishing is filling with AI-written output faster than it can be checked, and that one signal reviewers relied on has inverted. Two of its findings bear directly on the contract this repository exists to keep, so this note tests the repository against them rather than quoting them.*

---

## The two findings worth testing against

**1. Keeping the failures is worth money.** Eli Lilly is building **LillyPod** to apply deep learning to a **150-year archive "replete with unpublished failures as well as published successes"**, on the stated reasoning that this "should yield insights unavailable to those relying solely on the published scientific literature, **with its bias towards what worked**". A pharma company is spending on the negative results the literature threw away.

The same point arrives from the mathematics side of the package: a model can eventually generate a proof, but much of the insight sits in the attempts that failed, and "the complement only works if we get to see them".

**2. Presentation no longer carries information about substance.** A study in *Science* (December, as reported) found that **before large language models the sophistication of writing in a paper was generally a good guide to its scientific quality, and that the relationship has now reversed** — models "easily mimic the jargon, tone and style that signal expertise in a field". Around it: fabricated citations up **12-fold** between 2023 and early 2026 to roughly **0.5%** across 2.5m biomedical papers; about **90%** of papers in pay-to-publish journals AI-generated; submissions to one machine-learning conference **doubled in a year to 24,000**; and two preprints describing a **fictional eye disease** were being discussed as real by chatbots within weeks.

Taken together they describe a record that grows faster than it can be checked and is biased toward what worked. That is the failure mode [`tasks/chronos/`](../../tasks/chronos/) was built against, so the question is whether the build holds.

## What this repository already commits to

From [`tasks/chronos/README.md`](../../tasks/chronos/README.md), unchanged by this note:

> **Evidence is append-only.** Once a script run produces an evidence directory, it is not modified afterward. Re-runs create new timestamped directories. **Aborted runs move to an `aborted/` subdirectory but are not deleted.**

> If a claim cannot be traced to an artefact under `chronos/`, **treat it as commentary, not measurement.**

The second is the rule that survives finding 2 intact: it never relied on how a claim was written. A well-phrased claim with no artefact under `chronos/` is commentary here by definition, which is the only filter that still works when fluency is free.

## The measurement

The question finding 1 asks of us: **does the published record carry its negative results, or only the ones that worked?** Counted from the index table in `tasks/chronos/README.md`, today:

| | |
|---|---|
| Index rows (experiments, incidents, watcher runs) | **52** |
| Rows whose status records a **refuted, failed or null headline outcome** | **14** |
| Experiment directories carrying an `aborted/` subdirectory | **4** (exp_036, exp_037, exp_050, exp_051) |

The 14, with their statuses as written in the index:

- **Exp 015** — Attempted, not resolved
- **Exp 024** — Inconclusive
- **Exp 036** — Gate failed; stopped and published
- **Exp 037** — Stopped — budget (plan STOP by rule)
- **Exp 038** — H1 refuted (ceiling-limited)
- **Exp 040** — not robust (gain at Q4, none at Q8)
- **Exp 041** — no gate earns its place
- **Exp 046** — no gain, the small model reads the original
- **Exp 050** — H1 safety fail, H2 usefulness fail
- **Exp 051** — H3 fail (generation not faster), H4 fail
- **Exp 052** — H1 fail (MLX 10% slower)
- **Exp 053** — H1 fail, H2 fail (keep Q4_K_M)
- **Exp 055** — all five checks fail
- **Exp 057** — H1/H2 fail as registered, cause named

Method, so the number can be disputed: the status cell of each index row is matched against the markers *refuted, H‹n› fail, gate failed, no gain, not robust, inconclusive, not resolved, no gate earns, stopped, all five fail*. Rows with a mixed outcome (some hypotheses passing, some failing) are counted here as negative, because a registered hypothesis that failed is the thing at issue. Matches on "false refusals", "false spam" and "failover" are excluded — those are good results about false-positive rates, not failures. Reproduce it with the script in [`count.py`](./count.py).

**Reading:** 14 of 52, with four aborted-run directories kept on disk, is a record that publishes its negatives. The rule is operating, not merely written down. That is the claim this note makes, and the number is the basis — not that the rate is right, since there is no target rate and a programme could reach any share by choosing easier or harder questions.

## What the check found that is not fine

**The stated cadence does not match the record.** [`log/README.md`](../../log/README.md) opens "**One line a day** of building local AI tools on one Mac mini". [`log/2026-10.md`](../../log/2026-10.md) holds **five dated entries** for October (19th, 20th, 21st, 24th, 28th). Nothing is fabricated — days with nothing built are simply absent, which is the honest failure — but a reader cannot tell a day with no work from a day whose work was not written up, and the header invites them to assume neither happened. This is the same shape as the bias in finding 1, one level down: the record of the work is selected by whether there was something to report.

The package names the mechanism precisely, from the publishing side: **"AI is just the accelerant. The kindling is the publish-or-perish culture"** — AI-assisted submissions rose fastest at the institutions that reward raw output most directly. A daily-cadence promise is a quantity measure, and the way a quantity measure fails is silent omission on the thin days.

Two options, both honest, and the choice is not this note's to make: describe the cadence as it is ("entries on the days something changed"), or keep the daily promise and allow a null entry — *nothing built, here is why* — which makes a thin week visible instead of invisible. The second produces a more useful record and is the harder habit.

## What this note does not claim

- **The three external studies are cited by result, without method.** The *Science* reversal is the most load-bearing of them and the least specified in the reporting: no coefficient, sample or field breakdown. It is corroboration for a rule already in force here, not the reason for it.
- **LillyPod has produced no published result.** It is a stated plan with a stated rationale; the rationale is the part used here.
- **14 of 52 is a count of how statuses are written, not an audit of the experiments.** A status could be honestly worded and the underlying run still wrong. The artefacts are in the repository for anyone who wants to check the stronger claim.
- **Nothing here measures the selection one level up**: which questions got asked. A programme that only registers hypotheses it can verify cheaply will show a healthy negative rate and still have avoided the hard questions — and that selection is invisible inside any of these numbers. The honest statement of it belongs in the write-ups, not in a ratio.
