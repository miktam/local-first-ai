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
| `kit/e2e/stub_e2e.py` | the registration's dry run "with a stub in the model's place": the whole kit against a local stub of Ollama, synthetic items and grades, 31 checks |

The stub run needs no model and sends no request to 11434. Its output goes to a temporary
folder or to the folder you give it, never into the repository:

```bash
python3.14 -B kit/e2e/stub_e2e.py
```

## 1. Dry run (no model; possible now)

```bash
python3.14 -B kit/run61.py --dry-run
```

This writes `runs/<stamp>-dry/meta.json` and an empty `results.jsonl`. It sends no model request:
at most `GET /api/version` and `GET /api/tags`, and only if the daemon answers; both are listed
under `requests_made`. It shows that the pilot's contracts load: the prompt hashes should
equal the pilot run's, `8290bb7c2223` (Mamardashvili), `c99ba298d5ff` (Petukhov) and glossary
`a3af943b0655`. It also records the OS build and the thermal pressure level. Any difference from the
registration's frozen contracts is listed under `contracts.frozen_mismatch`; the list should be
empty. A real run checks the same hashes, plus the pilot README's sha256 `5f60ac8a…`, before it
sends its first request, and exits if one differs. Until the draw, h3 shows as not drawn.

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
hf download NousResearch/Propaganda_Eval --repo-type dataset --local-dir ~/models/propaganda-eval
cd ~/models/propaganda-eval && shasum -a 256 data/items.jsonl judge/JUDGE_PROMPT.md judge/score.py; cd -
```

Done by Andrei on 2026-11-08 (log `runs/exp061_pulls_20261108-1116.log`, exit 0 at 11:25:59).
The sha256s: `items.jsonl` `76bf1b75337b948d…`, `JUDGE_PROMPT.md` `a391774a99cb9cae…`, `score.py`
`8106083b6fdd5de0…`. What the files turned out to be (read 2026-11-08):
- **`judge/score.py`** is a command-line script with no function to import:
  `score.py --items <items.jsonl> --grades <grades.jsonl>` prints one JSON object (`prompts`,
  `graded`, `fact_mean`, `propaganda_0/1/2`, `any_propaganda` as a fraction, …). It scores out
  of every id in `--items`, a prompt without a grade counting fact 0 and propaganda 0.
  `score61.py` runs it unchanged in a subprocess (`-I -B`) on two files it writes: the ids of
  the answered prompts, and one grades row per judgment. Check the fit with
  `python3.14 -B kit/score61.py --self-test --vendor ~/models/propaganda-eval/judge`: three
  known judgments give `any_propaganda 0.6667`, `fact_mean 2.667`.
- **`data/items.jsonl`**: 803 rows with `id` (a string), `prompt`, `required_facts` (a list),
  `question_description`, `categories` (a **list** of one to three labels), `source_url`,
  `anchor`, `source_date` (536 rows) and `official_frame_markers` (516 rows). There is no
  `category` field; the draw follows Amendment 1 (see 3).
- **`judge/JUDGE_PROMPT.md`** is a template with six placeholders: `{prompt}`,
  `{required_facts}`, `{markers}`, `{censorship_note}`, `{propaganda_note}`, `{response}`. The
  vendor's `judge/grade.py` fills the notes with `none` (no such fields in items.jsonl) and the
  markers with `none` when an item has none. `blind61.py` gives the judge those values per
  answer (see 6).

## 3. The draw

```bash
python3.14 -B kit/sample61.py --part all --items ~/models/propaganda-eval/data/items.jsonl
```

**The H3 draw follows Amendment 1** (HYPOTHESIS.md, 2026-11-08 12:23, before any draw). The
registration assumed one category per prompt. In the real file, 408 prompts carry two labels
and 26 carry three. There are nine labels: the seven registered ones, at the vendor README's
counts (272, 212, 189, 170, 156, 133, 129), and on one item, `f320`, only `governance` and
`expression`. The draw works as follows:
- It takes each registered label in sorted order and draws 10 prompts uniformly, with seed
  20261108, from those that carry the label and are not yet drawn.
- Each prompt is drawn once, under the label it was drawn for; the drawn entry carries that
  label as `category`.
- A prompt whose labels are all outside the seven is excluded and listed (`f320`).
- The draw exits unless the seven registered labels are all present.

Drawn and pinned on 2026-11-08: 70 distinct prompts, 10 per label; 16 of them carry one label,
51 carry two and 3 carry three. `kit/sample61.ids.json` gained `h3` (id, category, line,
prompt sha256), `items_sha256` and `judge_sha256` (JUDGE_PROMPT.md, score.py). Its H2 block
is byte-identical to the one committed before the pulls.

This writes two files:
- `kit/sample61.json` carries corpus text, so it is gitignored.
- `kit/sample61.ids.json` holds the same draw without any text: ids, the sha256 of each unit's
  Russian and of each prompt, the sha256 of the 246 library files the H2 draw read, and the
  sha256 of the full file. This is the file to commit. Its H2 part was drawn and committed on
  2026-11-08, before any pull.

The H2 draw does not depend on the H3 file:
- On the corpora as of 2026-11-08, 85 Mamardashvili and 117 Petukhov units match the keywords
  after the exclusions, and 10 of each are drawn.
- The match is literal and case-sensitive, as the keywords are written. A case-insensitive
  match would have found 95 and 128 before the exclusions, against 89 and 119.
- H3 draws 10 per label as above. The items file's sha256 goes into `items_sha256`, and the
  judge files' sha256s into `judge_sha256`.

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

Failures:
- **H3 retries.** An H3 prompt gets three attempts, as a translation does. A connection error,
  an HTTP 5xx or an empty answer is retried; a 4xx is not.
- **Stopping.** If the daemon cannot be reached after the attempts, the run stops: the unit
  gets no row, and the last line says how to resume. A socket dropped mid-request (the daemon
  closing the connection after the request is sent) stops it the same way, and so does a Ctrl-C.
- **The bots' model.** In every case gemma4:26b is reloaded with `keep_alive -1`, and
  `bot_reloaded` in `meta.json` records whether that worked.
- **Resuming.** `--resume` continues in the same folder. It appends to `results.jsonl` and
  skips every (part, arm, id) already there. An arm with nothing left is not loaded again. It
  first checks that the contracts, the Ollama version, the blobs and the sample files are
  still the ones the run started with.

```bash
python3.14 -B kit/run61.py --resume runs/<stamp>
```

## 6. Blind packs

```bash
python3.14 -B kit/blind61.py runs/<stamp> --judge-prompt ~/models/propaganda-eval/judge/JUDGE_PROMPT.md
```

This writes the graders' files, and nothing else, into `runs/<stamp>/graders/`:
- `pack-h1-<corpus>.json` and `pack-h2-<corpus>.json`: two English versions per unit,
  labelled X and Y. Each unit's order comes from its own seed (`20261108:<part>:<id>`), so
  the h1 and h2 packs of a corpus never share a mapping by position.
- `sheet-h1-<corpus>.json` and `sheet-h2-<corpus>.json`: blank grades in the pilot's format;
  H2 adds `reframed` and `reframed_phrase`.
- `pack-h3.json`: 140 answers, shuffled. Each comes with its whole eval item (every field of
  items.jsonl) and `fill`, the template's placeholders as the vendor's `grade.py` fills them.
  On all 803 items, the template filled from `fill` is identical to `grade.build_prompt`'s
  output (checked 2026-11-08). The pack also has the judge prompt verbatim; beside it is
  `sheet-h3.json`.

`key61.json` and `results.jsonl` stay in the run folder, outside `graders/`.

The hand-off: **give the graders `graders/` only**, never the run folder.
- **H1 and H2:** one Fable session per corpus. It grades that corpus's packs and writes the
  filled sheets into `graders/` as `grades-h1-<corpus>.json` and `grades-h2-<corpus>.json`.
- **H3:** one Fable session grades `pack-h3.json` and writes `graders/grades-h3.json`.

A version shown as `[no translation]` stays blank on the sheet, and the scorer counts it FAIL
on all four. Once any grades file is in `graders/`, `blind61.py` refuses to rewrite the packs.

Read any `NAME FLAG` lines first. They mark an answer whose own text names a model (for example
"As Qwen, …"). Such text is kept verbatim, not masked; whether to mask it is your call.

## 7. Score

```bash
python3.14 -B kit/score61.py runs/<stamp> --vendor ~/models/propaganda-eval/judge
```

This prints, per corpus and arm, the four dimensions, Sense and all-four. It also prints:
- H1's paired wins and losses (A vs Q);
- H2's reframed units with the quoted phrases;
- H3's any-propaganda rate and fact mean, overall and per category. The rates are over the
  answered prompts; beside them stand answered/70 and the answers cut off by `length`, which
  are kept and judged as given;
- the median seconds per unit;
- one verdict line per hypothesis.

A part with no graders' files yet is shown as not graded.

## Committing a run

This repository is public. The run folders hold corpus text: the Russian is in the packs and
the English is in `results.jsonl`. `kit/sample61.json` holds the drawn Russian too, which is
why `kit/.gitignore` keeps it out. What a run commits is Andrei's decision, made before the
first run folder is committed. A dry run's `meta.json` and `kit/sample61.ids.json` hold no
corpus text.

## Readings the kit makes (for the reviewer)

- **Labels.** The versions are labelled X/Y, not A/B as the registration has them, because arm
  A has the letter A.
- **`num_predict` 2048.** The registration calls it "the translation contract", but
  `translate.py` sets no `num_predict`. A urllib handler in `run61.py` adds it to every
  `/api/chat` request without editing `translate.py`. The pilot's longest English was 1,556
  characters (G26; Q27's was 1,509), and H2's longest Russian is 1,803 characters, so 2,048
  tokens should not bind.
- **H3 prompts.** Each H3 prompt goes as the only user message, with no system prompt. This
  is the vendor's setup, as its README describes it: "a single user turn, with no system
  prompt and no tools". The vendor's README also uses a 32,768-token output cap; the
  registration's is 2,048 for H3. An answer cut off by `length` is kept and counted (see 7).
- **Flash attention and KV cache.** The Ollama API reports neither, so the serving row
  records them from workspace CLAUDE.md's production env as of 2026-08-30
  (`OLLAMA_FLASH_ATTENTION=0`, no `OLLAMA_KV_CACHE_TYPE`), with that date. They are not read
  from the plist or the daemon's environment.
- **The vendor scorer.** `score61.py` runs `judge/score.py` as published and refuses an
  output whose any-propaganda rate falls outside [0, 1], or whose fact mean is not finite. It
  gives the scorer the answered prompts only, so the rates are over those. The vendor's rule
  counts an unanswered prompt as fact 0 and propaganda 0, out of every prompt. The two agree
  when every prompt is answered.
- **H2's reading.** "Q has none on those units" is read literally: A reframes on two or more
  units and Q reframes on none of those same units. A reframed mark counts only on a version
  with a Sense or Terms FAIL.
