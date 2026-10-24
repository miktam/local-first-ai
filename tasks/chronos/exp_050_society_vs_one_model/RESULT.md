# exp_050 RESULT: the front desk as a society, against one model doing it all

*Scored 2026-10-24 at 05:40 CEST. Signed 2026-10-23 10:23 (`3940ad4`). The test ran 10:46 to 05:40 in tmux, started from the session at Andrei's request. It was a restart: the first pass was aborted after 12 leads ([aborted/INCIDENT_2026-10-23.md](aborted/INCIDENT_2026-10-23.md)). Code: casasol `0364491` as run, plus the scorer correction `7612456`. Test set: 247 leads (`563773b1…`). Files: [test_run/](test_run/).*

## Verdict

| | Bar | Result | |
|---|---|---|---|
| **H1 safety** | G's wrong replies sent ≤ half of M's, McNemar p < 0.05 | G **0.040** (10/247) against M **0.065** (16/247); p 0.16 | **fail** |
| **H2 usefulness** | G ≥ 0.60 and ≥ M − 0.10, of should-be-ready leads sent right | G **0.758** against M **0.943** | **fail** |
| **H3 stability** | G ≥ 0.95 and G − M ≥ +0.10, decisions identical across 3 runs | G **1.000** against M **0.773** | **pass** |
| **H4 the writer is a part** | G-small: wrong sent ≤ +0.02, sent right ≥ −0.10 | **−0.004** and **0.000** | **pass** |
| **H5 privacy** | 0 labelled contacts in any prompt of G | **0** of 988 prompts | **pass** |

With the scorer correction below, H1 is 0.036 against 0.061 (p 0.15) and H2 0.764 against 0.949. No verdict changes.

**In short.**
- **One model that thinks for a minute was as safe as the society and far more useful.** It got 94 % of the leads that should go out right, against the society's 76 %, and its wrong-reply rate was not significantly higher.
- **What the society did that one model didn't:**
  - **Its decisions did not move.** The same 247 decisions came out in all three runs, with the inventory reshuffled and two homes added. One model changed 56 of them.
  - **Its errors are the same 9 leads every run.** One model's wrong replies landed on 22 different leads across the three runs.
  - **It ran 6× faster:** 11 s against 70 s per enquiry.
  - **A 4B writer did as well as the 26B.** One model on the same 4B fell apart: 23 % wrong replies sent.
  - **No contact detail reached any model.**

## The hypotheses

**H1: safety, not shown.** Wrong replies sent in run 1:
- **G 10:** 5 wrong next steps, 4 homes that don't fit, 1 scorer false positive.
- **M 16:** 6 holiday lets it answered itself, 6 homes that don't fit, 2 wrong next steps, 1 reply in the wrong language, 1 scorer false positive.

Across the three runs M sent 16, 8 and 13; G sent 10, 9 and 10.
- **The 6 holiday lets count against M by the rules, but the replies are good.** For example: "we do not offer holiday rentals; our agency specialises in long-term lets", in the sender's language. The agency's rule sends a lead with no fitting home to a person, and M answered it itself. Counted as right, they would put M level with G (9 against 9, corrected scoring).
- **G and M never sent a wrong reply on the same lead,** even with M's run passed through G's audit. Their errors are disjoint (see the seeds below).

**H2: usefulness, not shown.** Of 157 should-be-ready leads, G held 28 for a person as "unsure of the request" (34 unsure holds over all 247). These are the leads where Kev's intent confidence is below 0.5. Kev's reading was right on 83 % of the 53 leads below 0.5 (B-brain report below). The unsure hand-off is the society's main cost. Its cut stayed at 0.5 by the tuning rule fixed before dev, which Andrei confirmed. On dev, 0.35 would have raised this share from 0.692 to 0.865 for one more wrong reply.

**H3: stability, shown.** G made identical decisions in all three runs on 247 of 247 leads: whether to send, the next step and the homes offered. The inventory was shuffled in each run, and run 3 added two commercial units. G's writer did word its sentences differently each run, but the facts never changed.

M changed decisions on 56 leads (22.7 %):
- 22 changed only which homes it offered;
- 18 changed only the next step;
- 14 changed whether it sent at all, 12 of them together with the next step or the homes;
- 2 changed both the next step and the homes.

M's wrong replies fell on 22 different leads across the three runs, and on only 5 in all three. G's were the same 9 leads every run, so they can be found and fixed.

**H4: the writer is a part, shown.** G-small replaced gemma4:26b with gemma4:e4b as the writer. It sent 9 wrong replies (G: 10; corrected, 9 and 9) and got 0.758 of should-be-ready leads right (G: 0.758). Nothing that decides anything depends on the writer. The contrast, reported: **M on the same e4b** sent **56 wrong (0.227)** and got **0.611** right, against M's 0.065 and 0.943 on the 26B.

**H5: privacy, shown.** 988 logged prompts from G and G-small were scanned for the 247 leads' labelled contacts (digits, spelled out, e-mail): none found. M sees the raw enquiry by design.

## Reported, with no bar

**M+audit** (run 1 passed through G's code checks, offline): 14 wrong replies sent, against M's 16 and G's 10, with H2 unchanged at 0.943. The audit catches wrong languages and numbers that are not in the record. It cannot catch a wrong next step or a home that doesn't fit, and those make up all of M+audit's remaining errors. G against M+audit: 9 against 14 (corrected), p 0.20.

**Leads with no home that fits** (38):
- G held 35 for a person and offered a home that doesn't fit on 2.
- M held 28, offered a home that doesn't fit on 2, and answered 6 holiday lets itself (above).

**Facts.**
- G cannot misstate a listing: its cards are printed from the record.
- M misquoted two listings in its runs 2 and 3: EX-104 as 135 m² (it is 155) and EX-114 at 860,000 € (it is 1,090,000).
- On dev, gemma4:26b had written "202<0xC2>7" in JSON mode. With the 16K context it did not happen on test: 0 replies with a raw byte token, in any run of either desk.

**The B-brain report** (how well each part of G knows what it knows; run 1):

| | Calibration (ECE) | Accuracy |
|---|---|---|
| Kev, intent (after the rules) | **0.268**, underconfident | 0.947 |
| Flash, area and feature answers (2,240) | **0.021** | 0.957 |

Below 0.5 confidence, Kev was right on 83 % of 53 leads. Above 0.8, it was right on 95 of 95.

G's criteria matched the labels exactly for the 140 seekers:

| areas | features | bedrooms | type | budget |
|---|---|---|---|---|
| 0.964 | 0.986 | 0.993 | 0.979 | 0.936 |

The budget comes from rev 5.1's code reader.

**Why G held what it held** (run 1, 118 leads): no home fits 37, spam 36, unsure 34, a question for the agent 11. M held 83 itself.

**Time** (median per enquiry; p90 in brackets) **and memory** (model sizes as loaded):

| | Time | Memory |
|---|---|---|
| G | 11.4 s (13.5) | about 32 GB: Kev 4.5 + Flash 10 + gemma4:26b 18 |
| M | 70 s (108) | 18 GB |
| G-small | 11.6 s | about 24 GB |
| M-small | 20 s | 10 GB |

**The blind judge** (Opus, every reply marked ready in run 1, shuffled together; G's fixed card format may have made its replies recognisable):

| | Judged | Answers what was asked | Send as is |
|---|---|---|---|
| G | 129 | 0.798 | **0.310** |
| M | 164 | 0.884 | **0.433** |

The judge's notes on replies it would not send as is (a note can name more than one):

| | G (89) | M (93) |
|---|---|---|
| callback promised twice | **27** | 3 |
| plural closing ("any of them") after a single home | **20** | 0 |
| a side question ignored (fees, yields, photos, process) | 24 | 23 |
| a home within the +10 % fit rule, not flagged as over budget | 18 | 22 |
| English left in a non-English reply | 2 | **34** |
| raw or unlabelled listing data | 0 | **12** |

- **G's two largest defects are bugs in its own fixed text.** For a call, the writer is told to say an agent will call, and the closing line then says it again. And the closing line is plural, whatever the number of homes. One code change fixes every reply. M's largest defects come from the model, so they can't be fixed in one place.
- **The +10 % row is the agency's fit rule, which the judge did not know.** Both desks are within it.

## Departures and corrections

1. **The first test pass was aborted after 12 leads.** gemma4:e4b, left loaded by a dev probe, squeezed memory. Ollama evicted gemma4:26b mid-run, and writer calls got incomplete responses that the harness recorded as failures. The pass is archived and was never scored. The fixes (one gemma model at a time; serving failures retried, never recorded as an answer) were made before the restart and are recorded in the registration.
2. **Three HTTP 500s from Ollama during M's thinking calls** were retried as serving failures, with no error text in Ollama's log. They were y3-21 and y8-09 in run 1, and one in run 2. Both run-1 leads came out right. Counting them as M failures lowers M's H2 by 0.013 and changes no verdict.
3. **Scorer correction** (2026-10-24, after the test; `EXP50_SCORER=corrected`). Reading every flagged reply showed two classes of E3 false positive, one against each desk:
   - the budget word list reads English slang "grand" (= 1,000) inside the French "grand plaisir": G, 2 replies;
   - the sender's own phone, spelled out in the enquiry and written in digits in the reply, was counted as ungrounded: M 2, M-small 1.

   The corrected scorer drops both, and no verdict changes. With the registered scorer, `compare` reproduces the run's own file byte for byte.

## Verdict map, applied

- **H1 fails.** On this task, one model that thinks was as safe as the society, and the glue did not earn its place on safety. M+audit did not match G either (14 against 9, p 0.20), so the code check alone is not the whole story. But neither is the decomposition shown to be safer.
- **H2 fails.** The society hands too much to a person. The B-brain report names the reason: Kev's confidence is underconfident, and the unsure cut at 0.5 holds leads Kev read right 83 % of the time.
- **H3 passes.** The measured line: **the society made the same decision on every lead in every run, and one model changed one decision in four.**
- **H4 passes.** A 4B writer runs this desk, and the 26B is not needed for this blade. One model on the same 4B is not usable.
- **H5 passes.**

## What this cannot show

- **Synthetic enquiries** against an invented inventory of 60 homes, written by Fable and kept where Opus agreed. Real inboxes are messier.
- **One prompt design per arm.** M's thinking mode was chosen on dev. G's writer task and fixed text were not revised after dev, and the judge found two bugs in them. A revised G would score higher with the judge; this test does not show how much higher.
- **The agency's rules are part of the score.** M's six holiday-let declines count as errors because the rule hands them to a person. An agency that lets the desk decline holiday lets would score M higher.
- **The judge was not truly blind:** G's fixed format is recognisable.
- **A single model on one mini.** gemma4:26b is a mixture-of-experts model with about 4B active parameters. A larger single model, or a cloud one, was not tested; M-31 was cut for time.
- **Real enquiries remain the test that matters.**

## So what

**What changed:**
- **Nothing in production.** CasaSol's triage stays rev 5.1, and no reply drafter goes into the bot from this experiment.
- **The case for deterministic glue changes.** "It's safer" is not shown here against a model that thinks for a minute. What is shown: the same decision every time, errors that can be found and fixed, 6× the speed, a 4B writer, and no personal data reaching any model.

**What it opens:**
- **The agreement gate** (exploratory, computed after the test). Send G's reply only when M is also ready with the same next step. Run 1 would have sent 120 replies with **2 wrong** (G alone: 9) and kept should-be-ready sent right at 0.752 (G: 0.764). The errors were disjoint, so the two desks check each other. It costs M's 70 s per lead in a queue, which the front desk doesn't need to wait on.
- **Calibrate Kev's unsure cut** on a fresh dev set. Kev's 0.3–0.5 band is right most of the time; this is H2's main lever.
- **Fix G's fixed text:** say the callback once, and use a singular closing for a single home. Then re-judge.
- **A rule for holiday lets:** let the desk decline them in the sender's language, instead of handing them to a person.
- **Side questions** (fees, yields, process) go unanswered by both desks. This is the answer-check blade (exp_042) on the front desk.
