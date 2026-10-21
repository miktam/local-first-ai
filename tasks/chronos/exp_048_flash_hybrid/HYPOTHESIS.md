# exp_048: the middle-sized decision model

*Written 2026-10-21, before any call on the test set. **Signed by Andrei 2026-10-21 06:19 CEST** ("sign"); text as committed in `20cdc55`, unchanged. Andrei: "let's try middle-sized model". Code: casasol `3f695d1` (`scripts/flash48.py` sha256 `77d18ef2…`); triage = rev 4.1 as frozen (`2acc7a1d…`). Model: Clef-Flash 9B Q8_0 (`Cloudflare_clef-flash-Q8_0.gguf`, about 9.7 GB, exp_041's file, prefix `55c222df`) on llama.cpp b11711, `/v1/systemone`, at :18448. No cloud model anywhere.*

## Question

exp_047 found that Clef 27B, behind rev 4.1's unchanged questions and rules, lifts "qualified" and timeline over the bars for showing them to an agent. It costs 13.5 s per enquiry and 29 GB. **Does Clef-Flash 9B, about a third of the size, keep most of that gain?** And if it doesn't on its own, does a split work: **Kev reads the request, Flash reads the facts?**

## Design

- **Enquiries:** the same 566 as exp_047 (`data/leads46/test.jsonl`, `a5f823d9…`). Every language, used before, nothing tuned on them.
- **flash:** rev 4.1 calling Clef-Flash. As in exp_047, the only change is the decision server; questions, code facts, rules and cuts are unchanged.
- **kev and clef27b:** exp_047's recorded test predictions on the same enquiries, same rev 4.1. Not re-run: Kev's probabilities move by about ±0.001 between identical calls, and decisions have held.
- **hybrid** (defined from the dev diagnosis below, and registered here, before the test). It is computed offline from the kev and flash arms' recorded answers, with no extra model call:
  - **Kev** answers the request side: its own request reading with probabilities, the business-pitch question and "wants to own".
  - **Flash** answers everything else: property type, timeline, financing, from abroad, qualified, hot, specific listing, call request, asks worth.
  - **rev 4.1's own `policy()`** turns the merged answers into the final request, urgency and next step.
  - **Self-check, run by the scorer:** recombining Kev's own answers reproduces every recorded Kev decision. Verified on dev.
- **Scoring:** as exp_047, with exp_043's "discard" read as "review".
- **Memory:** Flash needs about 10 GB, so it fits next to Ollama's models and Kev. Nothing is unloaded.
- **Guards** (`flash48.py run test`) refuse:
  - an unregistered set;
  - a `kev_knife.py` that isn't the frozen rev 4.1;
  - a server not serving the Flash GGUF;
  - an uncommitted runner;
  - missing exp_047 arms;
  - a second pass.

## As built: the dev shakedown (exp_047's 60 dev enquiries, 2026-10-21)

| | Kev | **Flash** | hybrid | Clef 27B |
|---|---|---|---|---|
| request | 0.933 | 0.867 (3 vs 7) | 0.933 | 0.967 |
| next step | 0.967 | 0.933 | 0.967 | 0.983 |
| qualified | 0.783 | 0.900 | 0.900 | 0.933 |
| timeline | 0.746 | 0.864 | 0.864 | 0.915 |
| property type | 0.883 | 0.967 | 0.967 | 0.967 |
| urgency within one step | 0.895 | 0.965 | 1.00 | 1.00 |
| genuine flagged as spam (49) | 0 | 2 | 0 | 0 |
| median seconds per enquiry | 1.7 | **3.7** | about 3.7 if both run at once | 13.2 |

- **All 8 of Flash's request errors are its own readings, not the rules.**
  - Buyers are read as investors (3).
  - Two genuine buyers acting through a lawyer or for a client are read as spam (X5-38, X4-39).
  - Flash reads the facts about as well as Clef 27B, and the request worse than Kev.
- **That diagnosis defines the hybrid.** H1–H5 were written into the scorer before the dev run. H6 and the hybrid were added after it, and before the test.

Dev files: [results/dev/](results/dev/).

## Hypotheses (test set, 566 enquiries, paired)

| | Measure | Bar |
|---|---|---|
| **H1 qualified showable** | Flash's "qualified" accuracy | ≥ 0.80 |
| **H2 timeline showable** | Flash's timeline accuracy | ≥ 0.75 |
| **H3 Flash no regression** | Flash − Kev, request and next step | ≥ −0.02 on each |
| **H4 Flash spam safety** | genuine flagged as spam; spam caught | ≤ Kev + 1; ≥ 0.85 |
| **H5 most of the gain** | Flash's share of Clef 27B's gain over Kev, on request, qualified and timeline | ≥ 0.5 on each |
| **H6 hybrid** | hybrid − Kev, request and next step; genuine flagged as spam | ≥ −0.02 on each; ≤ Kev + 1 |

Reported, with no bar:
- every field for all four arms;
- urgency;
- by language;
- Flash's confusions;
- McNemar Flash against Kev;
- speed.

## Verdict map

1. **H1–H5 pass → "The middle-sized model is enough."** Flash alone is the candidate: about a third of Clef 27B's memory, about 3.7 s per enquiry.
2. **Otherwise, if H1, H2 and H6 pass → "Split the work: Kev reads the request, Flash the facts."** The hybrid is the candidate: about 14 GB for both models. It takes about 3.7 s if the two calls run at once, 5.4 s one after the other.
3. **Otherwise, if H1 or H2 fails → "Only the 27B lifts the weak fields."**
4. **Otherwise → "Neither Flash alone nor the split holds up."**

Adoption, of any arm, stays Andrei's decision.

## What this cannot show

- **Synthetic enquiries,** labelled by two Fable readers who agreed. The ambiguous fifth was dropped.
- **Cuts tuned for Kev.**
- **One pass;** the comparison arms are exp_047's recorded passes.
- **The hybrid was designed after looking at dev.** The test set, not dev, decides H6.

## To run (after signing)

```bash
EXP=exp048 STEP=test; cd ~/REPOS/casasol && RUN=runs/${EXP}_${STEP}_$(date +%Y%m%d-%H%M)
cat > $RUN.sh <<'EOF'
$HOME/models/llama.cpp-b11711/build/bin/llama-server -m $HOME/models/exp041/clef-flash/Cloudflare_clef-flash-Q8_0.gguf --host 127.0.0.1 --port 18448 -c 16384 -b 8192 -ub 8192 > runs/exp048_flash_server_test.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:18448/health > /dev/null; do kill -0 $SRV || { echo "flash server exited"; exit 2; }; sleep 2; done
.venv/bin/python3 scripts/flash48.py run test
.venv/bin/python3 scripts/flash48.py score test > runs/exp048_test_score.json
EOF
tmux new-session -d -s $EXP-$STEP -c ~/REPOS/casasol "caffeinate -is zsh -e $RUN.sh 2>&1 | tee $RUN.log; echo \"[$EXP-$STEP] exit \${pipestatus[1]} at \$(date)\" | tee -a $RUN.log; exec zsh" && \
echo "attach: tmux attach -t $EXP-$STEP   (detach: Ctrl-b d)   log: tail -f ~/REPOS/casasol/$RUN.log"
```

About 35 min (566 × 3.7 s), plus loading the server and hashing the GGUF. The server stops when the script ends. An interrupted run resumes; a finished one is refused.
