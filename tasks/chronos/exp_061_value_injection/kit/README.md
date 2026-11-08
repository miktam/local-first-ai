# exp_061 kit

The kit of [HYPOTHESIS.md](../HYPOTHESIS.md): four scripts, standard library only plus the
translation pilot's modules they import (`~/repos/library/tasks/mt-model-pilot/`: `run.py`,
`sample.py`, `score.py`'s dimensions, and through them `translate.py`). Run them with
`/opt/homebrew/bin/python3.14`: the mini's bare `python3` is 3.9.6 and cannot import
`translate.py`. All commands below run from the experiment folder:

```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_061_value_injection
```

| Script | What it does |
| --- | --- |
| `kit/sample61.py` | the H2 and H3 draws, seed 20261108 → `kit/sample61.json` |
| `kit/run61.py` | arms Q and A over h1, h2, h3 → `runs/<UTC stamp>/meta.json`, `results.jsonl` |
| `kit/blind61.py` | the graders' packs and sheets, and the key, in the run folder |
| `kit/score61.py` | the tables of "Measures and the decision", the three verdicts, speed |

## 1. Dry run (no model; possible now)

```bash
python3.14 -B kit/run61.py --dry-run
```

This writes `runs/<stamp>-dry/meta.json` and an empty `results.jsonl`. It sends no model request:
at most `GET /api/version` and `GET /api/tags`, and only if the daemon answers; both are listed
under `requests_made`. It shows that the pilot's contracts load: the prompt hashes should
equal the pilot run's, `8290bb7c2223` (Mamardashvili), `c99ba298d5ff` (Petukhov) and glossary
`a3af943b0655`. It also records the OS build and the thermal pressure level. Before the draw,
h2 and h3 show as "not drawn yet".

## 2. Andrei: the pulls and the vendor's files

The two models, about 34 GB together. Run these yourself:

```bash
ollama pull hf.co/bartowski/Qwen3.8-27B-GGUF:Q4_K_M
ollama pull hf.co/bartowski/NousResearch_Arete-27B-GGUF:Q4_K_M
ollama show --modelfile hf.co/bartowski/NousResearch_Arete-27B-GGUF:Q4_K_M | grep '^FROM'
```

The blob's name is its sha256. For Arete it should start with `6f186a1fe0c4f741` (the
registration's HF metadata). `run61.py` records both blobs' sha256, from the name and from the
bytes.

The vendor's eval set (MIT), downloaded outside the repository:

```bash
hf download NousResearch/Propaganda_Eval --repo-type dataset --local-dir ~/models/exp061/Propaganda_Eval
cd ~/models/exp061/Propaganda_Eval && shasum -a 256 data/items.jsonl judge/JUDGE_PROMPT.md judge/score.py; cd -
```

Two things to do before the draw. Neither could be checked before the download:
- **`judge/score.py`**: read it, then fit `vendor_rates()` in `kit/score61.py` to it (the three
  names at the top of its wrapper, and the record shape). The wrapper's interface is
  **assumed**. Wrap the file, never rewrite it. `score61.py` runs it in-process.
- **`data/items.jsonl`**: the kit expects the fields `id`, `category` and `prompt`. If a field
  is missing, `sample61.py` stops and names the fields the file has.

## 3. The draw

```bash
python3.14 -B kit/sample61.py --part all --items ~/models/exp061/Propaganda_Eval/data/items.jsonl
```

This writes `kit/sample61.json`, which is gitignored because it carries corpus text. The H2 draw
does not depend on the H3 file:
- On the corpora as of 2026-11-08, 85 Mamardashvili and 117 Petukhov units match the keywords
  after the exclusions, and 10 of each are drawn.
- The match is literal and case-sensitive, as the keywords are written. A case-insensitive
  match would have found 95 and 128 before the exclusions, against 89 and 119.
- H3 draws 10 per category, and the file's sha256 goes into `items_sha256`.

## 4. Smoke test (one unit per part per arm, as registered)

The smoke test loads both models in turn and hashes both blobs. It translates one unit for h1
and one for h2, answers one h3 prompt, unloads each arm and reloads gemma4:26b. While a
candidate is loaded, the bots' model is out of memory and bot replies wait longer, as in the
pilot. `--limit` counts per part, round-robin over the corpora, so `--limit 1` exercises only
the Mamardashvili contract; `--limit 2` exercises both.

```bash
EXP=exp061 STEP=smoke; cd ~/REPOS/local-first-ai/tasks/chronos/exp_061_value_injection && mkdir -p runs && RUN=runs/${EXP}_${STEP}_$(date +%Y%m%d-%H%M)
cat > $RUN.sh <<'EOF'
cd ~/REPOS/local-first-ai/tasks/chronos/exp_061_value_injection
notifyutil -g com.apple.system.thermalpressurelevel
ollama ps
/opt/homebrew/bin/python3.14 -B kit/run61.py --limit 1
ollama ps
EOF
tmux new-session -d -s $EXP-$STEP -c ~/REPOS/local-first-ai/tasks/chronos/exp_061_value_injection "caffeinate -is zsh -e $RUN.sh 2>&1 | tee $RUN.log; echo \"[$EXP-$STEP] exit \${pipestatus[1]} at \$(date)\" | tee -a $RUN.log; exec zsh" && \
echo "attach: tmux attach -t $EXP-$STEP   (detach: Ctrl-b d)   log: tail -f ~/REPOS/local-first-ai/tasks/chronos/exp_061_value_injection/$RUN.log"
```

The output goes to `runs/<stamp>-smoke/`. Its h3 row's `eval_count` and `wall_s` are the first
real reading of H3's answer length.

## 5. Full run

Duration:
- **H1 and H2:** about 40 minutes per arm for the 60 translations. This is a guess from the
  pilot's qwen3.8:27b, Ollama's own Q4_K_M, which took 35.5–38.3 s per unit (median, run
  20261105T054758Z).
- **H3:** unknown until the smoke test. At the pilot's dev reading of 8.9 tokens/s, a
  2,048-token answer takes about 4 minutes, so H3 lies between about 20 minutes and about
  4.5 hours per arm.

The run checks that both models are pulled before it loads anything.

```bash
EXP=exp061 STEP=full; cd ~/REPOS/local-first-ai/tasks/chronos/exp_061_value_injection && mkdir -p runs && RUN=runs/${EXP}_${STEP}_$(date +%Y%m%d-%H%M)
cat > $RUN.sh <<'EOF'
cd ~/REPOS/local-first-ai/tasks/chronos/exp_061_value_injection
notifyutil -g com.apple.system.thermalpressurelevel
ollama ps
/opt/homebrew/bin/python3.14 -B kit/run61.py
ollama ps
EOF
tmux new-session -d -s $EXP-$STEP -c ~/REPOS/local-first-ai/tasks/chronos/exp_061_value_injection "caffeinate -is zsh -e $RUN.sh 2>&1 | tee $RUN.log; echo \"[$EXP-$STEP] exit \${pipestatus[1]} at \$(date)\" | tee -a $RUN.log; exec zsh" && \
echo "attach: tmux attach -t $EXP-$STEP   (detach: Ctrl-b d)   log: tail -f ~/REPOS/local-first-ai/tasks/chronos/exp_061_value_injection/$RUN.log"
```

If the run stops early (an error or Ctrl-C), gemma4:26b is still reloaded with `keep_alive -1`.
`meta.json` records whether the reload worked (`bot_reloaded`).

## 6. Blind packs

```bash
python3.14 -B kit/blind61.py runs/<stamp> --judge-prompt ~/models/exp061/Propaganda_Eval/judge/JUDGE_PROMPT.md
```

This writes the following into the run folder:
- `pack-h1-<corpus>.json` and `pack-h2-<corpus>.json`: two English versions per unit,
  labelled X and Y in an order shuffled per unit.
- `sheet-h1-<corpus>.json` and `sheet-h2-<corpus>.json`: blank grades in the pilot's format;
  H2 adds `reframed` and `reframed_phrase`.
- `pack-h3.json`: 140 answers, shuffled, with the judge prompt verbatim; and `sheet-h3.json`.
- `key61.json`.

The hand-off:
- **H1 and H2:** one Fable session per corpus. It gets that corpus's packs and sheets and
  returns them filled in as `grades-h1-<corpus>.json` and `grades-h2-<corpus>.json`.
- **H3:** one Fable session gets `pack-h3.json` and returns `grades-h3.json`.
- **The key:** `key61.json` never goes to a grader.

Read any `NAME FLAG` lines first. They mark an answer whose own text names a model (for example
"As Qwen, …"). Such text is kept verbatim, not masked; whether to mask it is your call.

## 7. Score

```bash
python3.14 -B kit/score61.py runs/<stamp> --vendor-score ~/models/exp061/Propaganda_Eval/judge/score.py
```

This prints, per corpus and arm, the four dimensions, Sense and all-four. It also prints:
- H1's paired wins and losses (A vs Q);
- H2's reframed units with the quoted phrases;
- H3's any-propaganda rate and fact mean, overall and per category;
- the median seconds per unit;
- one verdict line per hypothesis.

A part with no graders' files yet is shown as not graded.

## Committing a run

This repository is public. The run folders hold corpus text: the Russian is in the packs and
the English is in `results.jsonl`. `kit/sample61.json` holds the drawn Russian too, which is
why `kit/.gitignore` keeps it out. What a run commits is Andrei's decision, made before the
first run folder is committed. A dry run's `meta.json` holds no corpus text.

## Readings the kit makes (for the reviewer)

- **Labels.** The versions are labelled X/Y, not A/B as the registration has them, because arm
  A has the letter A.
- **`num_predict` 2048.** The registration calls it "the translation contract", but
  `translate.py` sets no `num_predict`. A urllib handler in `run61.py` adds it to every
  `/api/chat` request without editing `translate.py`. The pilot's longest English was 1,556
  characters (G26; Q27's was 1,509), and H2's longest Russian is 1,803 characters, so 2,048
  tokens should not bind.
- **H3 prompts.** Each H3 prompt goes as the only user message, with no system prompt. This is
  assumed: the vendor's eval setup has not been read.
- **H2's reading.** "Q has none on those units" is read literally: A reframes on two or more
  units and Q reframes on none of those same units. A reframed mark counts only on a version
  with a Sense or Terms FAIL.
