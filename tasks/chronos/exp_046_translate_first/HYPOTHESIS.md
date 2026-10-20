# exp_046: translate before triage

*Written 2026-10-20, before any call on the test set. **Signed by Andrei 2026-10-20 17:47 CEST** ("signed"); text as committed in `12b259b`, unchanged. Andrei chose option A ("go with A and E"), and "measure it, built to rule" for the invariant question. Code: casasol `5f0a1c4` (`scripts/translate46.py` sha256 `99aa0869…`); triage = rev 4 as frozen (`kev_knife_rev4.py` `b1940235…`); translator gemma4:26b, Ollama 0.40.2, digest `777b927e3b6b…`. No cloud model anywhere.*

## Question

Kev-4B's model card says English only, and Polish is its weakest language (0.83–0.86 in exp_044). **If a local model translates a non-English enquiry into English first, does Kev triage it better?**

## Design

- **Enquiries:** the non-English ones in exp_043's and exp_044's test sets, public copies. That is 425: 175 + 250, of which 68 are Polish, plus German, Spanish, French, Russian and Swedish (`data/leads46/test.jsonl`, sha256 `a5f823d9…`).
  - Both sets were used before, for exp_043's and exp_044's verdicts. They are reused for a different question, and nothing is tuned on them.
  - English enquiries are left out: they would be identical in both arms.
- **native:** rev 4 on the original text, the adopted triage.
- **translated:**
  1. Rev 4 masks the contacts.
  2. gemma4:26b translates the masked text into English, with thinking off, temperature 0 and seed 46.
  3. Kev answers its questions on the English.
  4. The code facts (language, budget, phone) and the agency's rules still come from the original. Only what Kev reads changes.
- **Invariant 1 of the deterministic-glue rules is broken by design:** Kev reads another model's prose. Andrei chose to measure it anyway. Adoption needs his explicit exception, the same one the voice path already relies on (Whisper → Kev). The other invariants hold:
  - every translation is written to disk (`translations.jsonl`);
  - the translator answers in a strict JSON contract;
  - code checks that every number of 3+ digits and every `[phone]`/`[email]` marker survives;
  - on any failure Kev reads the original, and the fallback is recorded;
  - prompts are small;
  - the topology is fixed.
- **Guards** (`translate46.py run test`) refuse:
  - an unregistered set;
  - a `kev_knife_rev4.py` that isn't frozen rev 4;
  - an uncommitted runner;
  - a different translator build;
  - a second pass of any arm.
- **Check** (`translate46.py check dev`): the runner's re-assembled triage, given the masked original, equals rev 4's own. Decisions are exact; probabilities agree within 0.005.

## As built: the dev shakedown (209 non-English dev enquiries, 2026-10-20)

**The bars below were written into the scorer before the dev run, and they are unchanged.**

| | native | translated |
|---|---|---|
| request | 0.923 | 0.895 (−0.028; 3 vs 9 discordant) |
| next step | 0.861 | 0.852 (−0.009) |
| genuine flagged as spam (175) · spam caught (34) | 1 · 32 | 2 · 31 |
| Polish request (33) | 0.848 | 0.879 |
| French · Spanish request (36 · 37) | 0.917 · 0.919 | 0.806 · 0.865 |

- **Translation itself worked:** 206 of 209 passed every check, with a median of 1.7 s.
  - The 3 fallbacks are the check doing its job: the translator *added* a `[phone]` or `[email]` marker where the source only said "this same number" or "my e-mail".
- **The losses come from Kev reading the English, not from bad translations.** Two faithful examples:
  - a Brussels lawyer buying for a client reads as a business pitch in English (X7-29, buy → spam);
  - a French question about agency fees reads as "sell" (W6-54).
- **Dev suggests H1 will fail.** It is registered anyway: 209 enquiries is small, and the test settles it on 425, with Polish as the open question.
- **Kev's probabilities jitter by about ±0.001 between identical back-to-back calls**, most likely llama.cpp's prompt cache. Decisions have not been seen to change (exp_045's text arm repeated 331 of 331), but a case sitting exactly on a cut could flip from this alone. The `check` command allows it.

Dev files: [results/dev/](results/dev/).

## Hypotheses (test set, 425 non-English enquiries, translated against native, paired)

| | Measure | Bar |
|---|---|---|
| **H1 gain** | request accuracy, translated − native | ≥ +0.02 and one-sided McNemar p < 0.05 |
| **H2 no harm** | next-step accuracy, translated − native | ≥ −0.01 |
| **H3 spam safety** | genuine enquiries flagged as spam; spam caught | translated ≤ native + 1; caught ≥ 0.85 |
| **H4 Polish** | Polish request accuracy (68), translated − native | ≥ +0.05 (no p: too few) |

Reported, with no bar:
- every language and both source sets;
- every enquiry whose request flips;
- the top confusions;
- translation fallbacks by reason;
- translation time.

## Verdict map

1. **H1, H2 and H3 pass → "Translate first helps."** It becomes an adoption candidate, which needs Andrei's invariant-1 exception, about 1.7 s more per enquiry, and gemma4:26b loaded (18 GB).
2. **H1 fails, H2 and H3 pass → "No gain: Kev reads the original."** No translation layer and no exception needed.
   - If H4 passes, add "Polish-only translation" as a seed for a fresh test.
3. **H2 or H3 fails → "Translation hurts."**

## What this cannot show

- **Synthetic enquiries,** labelled by two Fable readers who agreed (exp_043/044).
- **One translator, one prompt, one pass.**
- **Real enquiries mix languages and slang** more than these.

## To run (after signing), together with E's regression gate

```bash
EXP=exp046 STEP=test; cd ~/REPOS/casasol && RUN=runs/${EXP}_${STEP}_$(date +%Y%m%d-%H%M)
cat > $RUN.sh <<'EOF'
.venv/bin/python3 scripts/translate46.py run test --arm native
.venv/bin/python3 scripts/translate46.py run test --arm translated
.venv/bin/python3 scripts/translate46.py score test > runs/exp046_test_score.json
.venv/bin/python3 scripts/leads44.py regress scripts/kev_knife.py
EOF
tmux new-session -d -s $EXP-$STEP -c ~/REPOS/casasol "caffeinate -is zsh -e $RUN.sh 2>&1 | tee $RUN.log; echo \"[$EXP-$STEP] exit \${pipestatus[1]} at \$(date)\" | tee -a $RUN.log; exec zsh" && \
echo "attach: tmux attach -t $EXP-$STEP   (detach: Ctrl-b d)   log: tail -f ~/REPOS/casasol/$RUN.log"
```

It runs in this order:
1. the native arm, about 12 min;
2. the translated arm, about 25 min;
3. the score;
4. rev 4.1's regression gate on exp_044's sealed test, about 9 min. It exits 1 if rev 4.1 regresses, and the log ends with the exit code.

An interrupted arm resumes; a finished arm is refused.
