# exp_047: a bigger decision model behind the same rules

*Written 2026-10-20, before any call on the test set. **Signed by Andrei 2026-10-20 21:22 CEST** ("sign"); text as committed in `ee2cb65`, unchanged. Andrei chose option 1 ("go with 1"). Code: casasol `d2c4bfe` (`scripts/clef47.py` sha256 `2f9ef115…`); triage = rev 4.1 as frozen (`kev_knife.py` `2acc7a1d…`, the version live in the bot). Decision models: Kev-4B Q8_0 (the bot's service, :18442) and Clef 27B Q8_0 (`Cloudflare_clef-Q8_0.gguf`, exp_041's file, prefix `9fe953f0`) on llama.cpp b11711 (`9d1e036`), `/v1/systemone`, at :18447. No cloud model anywhere.*

## Question

**Is Kev-4B the ceiling, or are the rules?** Every triage result so far ran Kev-4B behind code rules. Two of its answers were too weak to show an agent: "qualified" (0.78 against a 0.80 bar) and timeline (0.73 against 0.75, exp_043). Here Clef 27B, about 7× larger, answers exactly the same questions. The same code turns its answers into an urgency and a next step, with the same cuts.

## Design

- **Enquiries:** all 566 of exp_043's and exp_044's test sets, public copies, every language (`data/leads46/test.jsonl`, `a5f823d9…`). Both sets were used before for other verdicts; nothing is tuned on them.
- **kev:** rev 4.1 calling Kev-4B, the triage live in the bot today.
- **clef:** the same file, rev 4.1, calling Clef 27B. The only change is the decision server (`KEV_URL`).
  - The cuts (business pitch 0.60, Kev's own spam 0.15, wants to own 0.65) and the 0.5 thresholds stay as tuned for Kev. That is the question: the same policy, a bigger model.
  - Re-tuning the cuts for Clef would be a separate test.
- **Scoring:** paired, enquiry by enquiry.
  - exp_043's labels call spam's next step "discard"; it is read as "review", the same decision under its current name.
  - That label went unconverted in exp_046's scorer, so its published next-step numbers are understated in both arms (0.873 / 0.868; 0.922 / 0.920 corrected). Its verdict is unchanged; the note goes into exp_046's records.
- **Memory:** Clef needs about 29 GB. The run script unloads production Ollama's three pinned models first (gemma4:26b, gemma4:e4b, Qwen3-4B; production Ollama isn't in use, and each reloads on its next request). It starts Clef's server, and stops it at the end.
- **Guards** (`clef47.py run test`) refuse:
  - an unregistered set;
  - a `kev_knife.py` that isn't the frozen rev 4.1;
  - a server not serving the Clef GGUF;
  - an uncommitted runner;
  - a second pass of any arm.

## As built: the dev shakedown (60 dev enquiries, an even spread, 2026-10-20)

**The bars below were in the scorer before the dev run, and they are unchanged.** There is no tuning in this experiment, so dev only checked the pipeline.

| | Kev | Clef |
|---|---|---|
| request | 0.933 | 0.967 (4 vs 2) |
| next step | 0.967 | 0.983 |
| **qualified** | 0.783 | **0.933** (9 vs 0) |
| **timeline** | 0.746 | **0.915** (12 vs 2) |
| property type | 0.883 | 0.967 |
| request, model's own reading before the rules | 0.867 | 0.967 |
| urgency within one step | 0.895 | 1.00 |
| genuine flagged as spam (49) · spam caught (11) | 0 · 10 | 0 · 10 |
| **median seconds per enquiry** | **1.7** | **13.2** |

- **Clef's yes/no answers are more decisive.** Its "qualified" scores sit near 0.1 or 0.95 where Kev's sit near 0.5.
- **Kev's probabilities jitter by about ±0.001 between identical calls** (exp_046). A case sitting exactly on a cut could flip from this alone.

Dev files: [results/dev/](results/dev/).

## Hypotheses (test set, 566 enquiries, clef against kev, paired)

| | Measure | Bar |
|---|---|---|
| **H1 qualified** | "qualified" accuracy, where labelled | Clef ≥ 0.80 and ≥ Kev + 0.05 |
| **H2 timeline** | timeline accuracy, where labelled | Clef ≥ 0.75 and ≥ Kev + 0.05 |
| **H3 no regression** | request and next step, Clef − Kev | ≥ −0.02 on each |
| **H4 spam safety** | genuine enquiries flagged as spam; spam caught | Clef ≤ Kev + 1; caught ≥ 0.85 |

Reported, with no bar:
- request (McNemar), property type, financing and from-abroad where stated, urgency within one step;
- the model's own request before the rules;
- by language and source set;
- top confusions, valuation vs sell among them;
- speed.

## Verdict map

1. **H1 or H2 passes, and H3 and H4 pass → "The bigger model lifts the weak fields."** That field, or both, becomes showable with Clef. Adoption is Andrei's decision, at about 8× the time per enquiry and 29 GB loaded, or as a hybrid that asks Clef only the weak questions, which would need its own test.
2. **H1 and H2 fail, H3 and H4 pass → "Kev is enough."** The weak fields are a question or rule problem, which rev 5 takes on.
3. **H3 or H4 fails → "Clef does worse under Kev's rules."** Re-tuning the cuts for Clef would be a separate test.

## What this cannot show

- **Synthetic enquiries,** labelled by two Fable readers who agreed. The ambiguous fifth was dropped.
- **Cuts tuned for Kev,** not Clef.
- **One pass per arm.**
- **Real inbox volume:** at 13 s per enquiry, a busy inbox needs a queue (SPECIALISTS.md's design), not instant replies.

## To run (after signing)

```bash
EXP=exp047 STEP=test; cd ~/REPOS/casasol && RUN=runs/${EXP}_${STEP}_$(date +%Y%m%d-%H%M)
cat > $RUN.sh <<'EOF'
.venv/bin/python3 scripts/clef47.py run test --arm kev
for m in gemma4:26b gemma4:e4b hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M; do ollama stop "$m" 2>/dev/null || true; done
$HOME/models/llama.cpp-b11711/build/bin/llama-server -m $HOME/models/exp041/clef-27b/Cloudflare_clef-Q8_0.gguf --host 127.0.0.1 --port 18447 -c 16384 -b 8192 -ub 8192 > runs/exp047_clef_server_test.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:18447/health > /dev/null; do kill -0 $SRV || { echo "clef server exited"; exit 2; }; sleep 2; done
.venv/bin/python3 scripts/clef47.py run test --arm clef
.venv/bin/python3 scripts/clef47.py score test > runs/exp047_test_score.json
EOF
tmux new-session -d -s $EXP-$STEP -c ~/REPOS/casasol "caffeinate -is zsh -e $RUN.sh 2>&1 | tee $RUN.log; echo \"[$EXP-$STEP] exit \${pipestatus[1]} at \$(date)\" | tee -a $RUN.log; exec zsh" && \
echo "attach: tmux attach -t $EXP-$STEP   (detach: Ctrl-b d)   log: tail -f ~/REPOS/casasol/$RUN.log"
```

It runs in this order:
1. the Kev arm, about 17 min;
2. unload Ollama's models, then start Clef's server, about 1–2 min to load;
3. the Clef arm, about 2 h 5 min (566 × 13 s);
4. the score.

The server stops when the script ends. An interrupted arm resumes; a finished arm is refused. **Total about 2 h 25 min.**
