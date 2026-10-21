# exp_049: lead triage rev 5 against rev 4.2, on a fresh set

*Written 2026-10-21, before any call on the test set. **Signed by Andrei 2026-10-21 12:53 CEST** ("sign"); text as committed in `e273857`, unchanged. Andrei: "let's start using Fable as we planned", after rev 4.2 went live. Code: casasol `bfcd0d6`:
- `scripts/leads49.py`;
- rev 4.2 = `kev_knife.py`, sha256 `5b79950f…`, live in the bot;
- rev 5 = `kev_knife_rev5.py`, sha256 `8d268660…`, frozen after dev tuning.

Both versions use Kev-4B (:18442) and Clef-Flash 9B (:18448). Data: Fable 5.1 writers, an Opus 5.5 blind checker, gemma4:26b as a third reader.*

## Question

Rev 4.2 splits the work: Kev-4B reads the request, Clef-Flash 9B the facts. Its recorded errors on exp_043/044 fall into four patterns:
- **timing given as a bare month** ("from 1 November", "in December", "for Easter"), the most common, because the model isn't told today's date;
- **owners who only want a price, read as sellers;**
- **genuine clients writing through an intermediary, read as spam;**
- pitches read as clients.

**Do rev 5's targeted fixes remove the first three, without making anything else worse, on fresh enquiries written to contain them?**

Rev 5 = rev 4.2 plus four changes, each tuned on dev only:

| | Change | Answered by |
|---|---|---|
| **F1** | an `on_behalf` question; at ≥ 0.8 it overturns a spam reading (to the best non-spam reading) | Flash |
| **F2** | a `price_only` question; at ≥ 0.7 it turns a "sell" reading into valuation | Flash |
| **F3** | the timeline question states today's date ("Today is 21 October 2026…", with the label rule's buckets) | Flash, as before |
| **F4** | the financing question asks "use a mortgage or loan, already approved or not?" (exp_047) | Flash, as before |

The request side (request, business pitch, wants to own) stays Kev's. The rules and cuts from rev 4.2 are unchanged.

## Data (as built, 2026-10-21)

- **Labels v3** ([LABELS.md](LABELS.md)): exp_044's rules, plus `on_behalf`, the timeline rule against an explicit date (21 October 2026; January 2027 counts as `later`), and financing as "uses a mortgage". The urgency scale and the action rules are unchanged.
- **Writers:** eight Fable writers, 70 each:
  - two English (one deliberately messy), Polish, Spanish, German, Swedish, French, Russian;
  - 40 standard enquiries per writer, with exp_044's quotas scaled;
  - three hard strata of 10 per writer, each **written from rev 4.2's recorded error patterns**, described, never copied ([writer_prompt.md](writer_prompt.md)):
    - `hard_intermediary`: 5 genuine clients writing through someone, 5 pitches from businesses that sound like them;
    - `hard_valuation`: 5 price-only owners, 5 decided sellers who also ask the price;
    - `hard_timing`: timing only as a month, date, season or holiday.
  - **Totals:** 560 enquiries, with no schema errors, quotas exact and **no realistic contact details**. The contact check is now digit-based, so Ofcom numbers pass in any spacing.
  - **Departures:** one writer (German) and one checker (batch 06) used a shell command to append their own output, which the brief forbade. Neither read any other file. The checker also ran an id-match check on its own two files.
- **Checker: Opus**, blind. That's a different model from the writers', unlike exp_044, where Fable checked Fable ([checker_prompt.md](checker_prompt.md)).
  - The agreement rule is as in exp_043/044: drop a lead on intent disagreement or the checker's doubt, and exclude any other field they disagree on.
  - **513 kept, 47 dropped**, all on doubt. There were no intent disagreements, and field agreement among kept leads was 0.94–1.00.
  - Kept by stratum: standard 279, and 78 in each hard stratum.
- **Split** (seed 49, by language × stratum): **dev 163** (`3c22399d…`), **test 350** (`de3e1710…`). Test has 188 standard and 54 in each hard stratum.
- **Andrei's read of 30 test leads** ([data/READ30_notes.md](data/READ30_notes.md)): two notes, both about the action rules (a call request; rule order), not about labelling under them. **Decision: keep the rules, reject no lead.** The call-request rule is a seed for after this experiment.
- **Third reader: gemma4:26b** (Google; neither the writers', the checker's nor the triage models' family). It labels every kept lead after the test pass. Reported only: it never drops a lead.

## As built: dev (163 enquiries)

The bars below were written into `leads49.py compare` before the dev runs, and they are unchanged.

| | rev 4.2 | rev 5, starting cuts (Kev, 0.6 / 0.6) | **rev 5, tuned (Flash, 0.8 / 0.7)** |
|---|---|---|---|
| request, standard (91) | 0.934 | 0.956 | 0.956 |
| request, intermediaries (24) | 0.917 | 0.958 | 0.958 |
| request, valuation stratum (24) | 0.875 | 0.917 | **1.000** |
| timeline, timing stratum (24) | 0.417 | **0.833** (F3) | 0.833 |
| timeline, standard | 0.789 | 0.889 | 0.889 |
| genuine flagged as spam · spam recall | 1 · 1.00 | 0 · 1.00 | 0 · 1.00 |
| request, all dev | 0.933 | 0.957 | **0.969** |

- **Tuning** (`leads49.py tune`, offline over both models' recorded answers):
  - grid: the source of each new question (Kev or Flash) × the cuts {off, 0.5–0.8} × the reverse rule;
  - choice: the best dev request accuracy with no more false spam than rev 4.2 and spam recall within 0.05; ties → the most cautious.
  - **Flash read both new questions better:** `on_behalf` 161/163 against Kev's 156, and `price_only` 24/24 on the valuation stratum against 18.
- **The frozen rev 5 reproduces the tuned offline decisions** on 8 of 8 dev enquiries re-run live.
- **Speed:** like rev 4.2, both models answer, so about 5.5 s per enquiry.

Dev files: [results/dev/](results/dev/).

## Hypotheses (test set, 350, rev 5 against rev 4.2, paired)

| | Measure | Bar |
|---|---|---|
| **H1 no regression** | standard stratum (188): request and next step, rev 5 − rev 4.2 | ≥ −0.02 on each |
| **H2 intermediaries** (F1) | genuine clients flagged as spam in `hard_intermediary`; all genuine flagged as spam; spam recall | at most half of rev 4.2's in the stratum (≤ rev 4.2's if it has fewer than 2); all ≤ rev 4.2's; recall ≥ 0.85 and ≥ rev 4.2's − 0.05 |
| **H3 valuation** (F2) | `hard_valuation` (54): request, rev 5 − rev 4.2 | ≥ +0.10 and one-sided McNemar p < 0.05 |
| **H4 timing** (F3) | `hard_timing` (54): timeline, rev 5 − rev 4.2; standard stratum timeline | ≥ +0.10 and p < 0.05; standard ≥ −0.02 |

Reported, with no bar:
- every field, all strata, by language;
- financing (F4) where stated;
- `on_behalf` accuracy;
- offline ablations of F1 and F2 on the recorded test answers (exact, since they are rules over recorded answers);
- the third reader's agreement with the kept labels, by field and stratum, and the accuracy of both versions on the leads where it agrees too.

## Verdict map

- **Each fix is adopted if its own hypothesis passes and H1 passes:** F1 ↔ H2, F2 ↔ H3, F3 ↔ H4.
  - A fix whose hypothesis fails is switched off, and the rest are adopted. F1 and F2 switch off exactly, by their cuts. F3 needs the old timeline question back.
  - F4 is adopted with F3, if financing accuracy where stated is no worse.
- **If H1 fails, rev 5 is not adopted.** The flips show what broke.
- **Adoption in the bot** runs through the regression gate on exp_044's sealed test, then a deploy. Both steps are Andrei's.

## What this cannot show

- **Synthetic enquiries, and hard strata written to contain the errors.** They show whether the fixes work on these patterns, not how often the patterns occur in an inbox.
- **One pass per version.** Kev's probabilities move by about ±0.001 between identical calls.
- **The cuts were tuned on 163 dev enquiries;** the test decides.
- **Real enquiries remain the test that matters.**

## To run (after signing)

```bash
EXP=exp049 STEP=test; cd ~/REPOS/casasol && RUN=runs/${EXP}_${STEP}_$(date +%Y%m%d-%H%M)
cat > $RUN.sh <<'EOF'
.venv/bin/python3 scripts/leads49.py run test rev42
.venv/bin/python3 scripts/leads49.py run test rev5
.venv/bin/python3 scripts/leads49.py compare test > runs/exp049_test_compare.json
.venv/bin/python3 scripts/leads49.py third
.venv/bin/python3 scripts/leads49.py third-score > runs/exp049_third_score.json
EOF
tmux new-session -d -s $EXP-$STEP -c ~/REPOS/casasol "caffeinate -is zsh -e $RUN.sh 2>&1 | tee $RUN.log; echo \"[$EXP-$STEP] exit \${pipestatus[1]} at \$(date)\" | tee -a $RUN.log; exec zsh" && \
echo "attach: tmux attach -t $EXP-$STEP   (detach: Ctrl-b d)   log: tail -f ~/REPOS/casasol/$RUN.log"
```

It runs in this order:
1. rev 4.2 on the 350 test enquiries, about 32 min;
2. rev 5, about 32 min;
3. the comparison;
4. the gemma4:26b third reader over all 513 kept leads, about 30–40 min;
5. its agreement.

**About 1 h 40 min in all.** Both decision services (Kev :18442, Flash :18448) must be up; the runner refuses otherwise. An interrupted step resumes; a finished test pass is refused.
