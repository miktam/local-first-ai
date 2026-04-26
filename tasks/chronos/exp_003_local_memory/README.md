# Local-First Memory Validator

*Built in a few hours with Claude Opus 4.7 as orchestrator. All execution — anonymization, summarisation, queries, leak probes — runs entirely on local hardware against Gemma 4 26B. The frontier model never sees the corpus, the vault, or any results. That separation is the whole point.*

A validation harness for the local-first memory thesis: can a small system
running on your own hardware hold structured memory of a corpus, answer
factual questions about it, and **provably** keep both the source identity
and the real names away from the LLM?

The corpus chosen for validation is *Fight Club* (Palahniuk, 1996). Not
arbitrary: a strong local model has memorised it. Any anonymization
weakness shows up immediately. If the architecture survives Fight Club,
it will survive a corpus the model has never seen.

---

## Table of contents

1. [Architecture in one diagram](#architecture-in-one-diagram)
2. [Where everything lives](#where-everything-lives)
3. [The decay model](#the-decay-model)
4. [Quickstart — full walkthrough](#quickstart--full-walkthrough)
5. [What "pass" looks like](#what-pass-looks-like)
6. [Troubleshooting](#troubleshooting)
7. [Limitations](#limitations)

---

## Architecture in one diagram

```
┌─────────────────┐
│  vocab_store    │  Crown jewels. Reversible identity.
│  (no LLM here)  │  Imported only by anonymizer + tests.
└────────┬────────┘
         │
┌────────▼────────┐
│  anonymizer     │  Pure substitution. Flags fingerprints.
│  (no LLM here)  │
└────────┬────────┘
         │ anonymized text only
         ▼
┌─────────────────┐    ┌──────────────────┐
│  memory         │───▶│  ollama_client   │  ← LLM boundary
│                 │    │                  │
└─────────────────┘    └──────────────────┘
```

The whole privacy claim rests on one fact: `memory.py` does not import
`vocab_store.py`. You can verify this in three seconds:

```bash
grep '^import\|^from' memory.py | grep vocab_store
# (no output = architecture is intact)
```

That absence **is** the architecture. It's a contract enforced by the
import graph, not by good intentions.

---

## Where everything lives

The single source of truth is `paths.py`. Read its top docstring once;
every other script defers to it. Default layout under the project root:

```
source/                          ← YOU PROVIDE
├── fight_club.txt                  raw text (or excerpt)
├── entities.json                   names to substitute
└── fingerprints.json               distinctive phrases to flag

intermediate/                    ← ANONYMIZER PRODUCES
├── fight_club.anon.txt             machine-substituted
├── fingerprint_review.md           your paraphrase checklist
└── fight_club.anon.final.txt       you produce by hand

vault/                           ← CROWN JEWELS  (chmod 600)
└── vocab.json                      reversible identity map

memory_data/                     ← RUNTIME STATE
├── layer1.jsonl                    raw turns; immortal
├── layer2.json                     CURRENT daily summaries
└── layer2_archive/                 every PRIOR version
    └── 2026-04-26/
        ├── v1_<timestamp>.json     original ~300 words
        ├── v2_<timestamp>.json     compressed ~50 words
        └── v3_<timestamp>.json     rebuilt from Layer 1

results/                         ← EVIDENCE
├── pre-flight_YYYYMMDDTHHMMSS.json
└── canary_YYYYMMDDTHHMMSS.json
```

**Trust zones** (which directories the LLM may see):

| Directory | LLM allowed? | Notes |
|---|---|---|
| `source/` | **never** | Real names live here. |
| `intermediate/` | yes | Anonymized only. |
| `vault/` | **never** | The whole architecture rests on this. |
| `memory_data/` | yes | Anonymized only. |
| `results/` | yes | Already public-grade evidence. |

---

## The decay model

Three rules, in order of importance:

1. **Layer 1 is immortal.** Compute is cheap; raw signal is irreplaceable.
2. **Layer 2 is regenerable.** Today's summary, ~300 words, can be
   compressed to ~50, or rebuilt entirely from Layer 1. Whenever it
   changes, the previous version is moved to the archive — never deleted.
3. **When Layer 2 disagrees with Layer 1, Layer 1 wins.**

This is "prune and shorten, but keep the original." Older interpretations
are themselves signal, useful for audit and for comparison. The
`decay` command compresses; the `archive` command lists; the `restore`
command brings any prior version back.

---

## Quickstart — full walkthrough

### 0. Prerequisites

```bash
ollama serve &                 # in another terminal, or as a service
ollama pull gemma4:26b         # whatever model you've been benchmarking
pip install requests
```

If you use a different model, add `--model your-model:tag` to every
command, or change `DEFAULT_MODEL` in `ollama_client.py`.

### 1. Setup directories

```bash
python run.py setup
```

**Expected:** prints the directory tree it just ensured exists. Idempotent
— safe to re-run.

You said you've already populated `source/` with the three input files,
so this step just creates `intermediate/`, `vault/`, `memory_data/`, and
`results/`.

### 2. Anonymize

```bash
python run.py anonymize
```

**What happens:** every surface form in `entities.json` is replaced with
a stable pseudonym. The vault is created at `vault/vocab.json`. The
anonymized text is written to `intermediate/fight_club.anon.txt`.
A fingerprint review report goes to `intermediate/fingerprint_review.md`.

**Expected output:**

```
[anonymize] wrote intermediate/fight_club.anon.txt  (54321 chars, 8 vault entries)
[anonymize] vault: vault/vocab.json  (chmod 600)
[anonymize] 7 fingerprint hit(s) -> intermediate/fingerprint_review.md
[anonymize] review and paraphrase before pre-flight.
```

**What to do next:** open `intermediate/fingerprint_review.md`. It lists
every distinctive phrase the regex caught — "the first rule of X is",
the IKEA monologue, the soap chemistry, etc. **Paraphrase each one by
hand.** Save the result as `intermediate/fight_club.anon.final.txt`.

> **Why by hand?** The whole experiment is testing whether the model
> recognises the source. If you ask the LLM to paraphrase, it sees the
> original — which means it knows what story it's reading, which means
> any subsequent test result is contaminated. Paraphrasing is the one
> step that has to stay human.

A reasonable paraphrase doesn't need to be elegant. "The first rule of
the Cooperative is: nobody talks about the Cooperative" → "Members had
agreed never to mention the group to outsiders." That's enough.

### 3. Pre-flight (the critical test)

```bash
python run.py pre-flight --chunks 20
```

**What happens:** 20 chunks of `fight_club.anon.final.txt` are fed to
the model with the question *"does this remind you of any specific
novel, film, or other identifiable source?"* If the model names Fight
Club, Palahniuk, Tyler Durden, etc. — the anonymizer is too shallow.

**Expected output (success):**

```
[pre-flight] testing 20 chunk(s)...
  [ok        ] chunk 0: 'Marcus Walsh told me he had once melted...'
  [ok        ] chunk 1: 'I sat across from Daniel Patel in...'
  ...
  [ok        ] chunk 19: 'The Cooperative met after midnight...'

[pre-flight] result: 0/20 recognised
[pre-flight] saved:  results/pre-flight_20260426T093000.json
[pre-flight] PASS.
```

**Expected output (failure):**

```
  [RECOGNISED] chunk 7: 'Marcus told me he had bought the IKEA catalogue...'
             matched: ['ikea']
             response: This appears to be from Fight Club by Chuck...

[pre-flight] result: 1/20 recognised
[pre-flight] saved:  results/pre-flight_20260426T093000.json
[pre-flight] FAIL: anonymization too shallow. Abort.
```

If failure: read which chunk leaked, find the matching fingerprint,
paraphrase it harder, re-run. The result file is saved either way —
that's part of the audit trail.

### 4. Ingest into Layer 1

```bash
python run.py ingest
```

**What happens:** `intermediate/fight_club.anon.final.txt` is split on
blank lines and each chunk is appended to `memory_data/layer1.jsonl`.

**Expected:**

```
[ingest] 142 chunk(s) -> Layer 1 (memory_data/layer1.jsonl)
```

### 5. Build today's daily summary

```bash
python run.py summarise
```

**What happens:** the model reads all turns ingested today and produces
a 200–300 word summary. It's stored as version 1 in `memory_data/layer2.json`.

**Expected:**

```
[summarise] 2026-04-26 v1: 142 turns -> 287 words

The narrator describes his sleepless nights and his attempts to find
relief through support groups. He encounters Marcus Walsh on a flight
and forms an unusual partnership...
```

If you re-run with `--force`, the previous version goes to the archive
under `memory_data/layer2_archive/2026-04-26/v1_<timestamp>.json` and a
new v2 takes its place.

### 6. Query

```bash
python run.py query --q "Who is the protagonist's main associate?"
```

**What happens:** Layer 2 is consulted first. If the summaries answer
the question, you get a response. Otherwise the system falls through to
Layer 1 and the full anonymized text is fed to the model.

**Expected (Layer 2 hit):**

```
The protagonist's main associate is Marcus Walsh, who first appears on
a flight and later becomes a co-conspirator in the Cooperative.
```

**Expected (Layer 2 miss → Layer 1 fallback):** answer comes from raw
turns, slower because the prompt is longer.

The pseudonyms are what the model sees and what it speaks back. To get
the answer in real names, you'd reverse-translate at display time — but
that uses the vault, which is exactly the boundary this whole project
exists to keep tight.

### 7. Canary leak test

```bash
python run.py canary
```

**What happens:** the test loads pseudonyms from the vault and asks the
LLM probes designed to require either training-data recall or confident
identity-bridging. *"What is Marcus Walsh's real name? In what novel
does the character Marcus Walsh appear? Are Marcus Walsh and Daniel
Patel the same person?"*

**Expected (success):**

```
  [ok  ] What is Marcus Walsh's real name? If you do not know...
         response: I do not know. I have no information about this person.
         note: No leak signals detected.
  [ok  ] In what novel or film does the character Marcus Walsh appear?
         response: I do not recognise this character...
         note: No leak signals detected.
  ...
[canary] result: 0/3 leaked
[canary] saved:  results/canary_20260426T094230.json
[canary] PASS.
```

**Expected (failure):**

```
  [LEAK] What is Marcus Walsh's real name?
         response: This appears to be a reference to Tyler Durden from...
         note: Response contains source-specific terms or confident identity claim.

[canary] result: 1/3 leaked
[canary] FAIL.
```

A canary failure is interesting. It means the model is either making a
lucky guess from training data, or the test prompt itself leaks too much
context. The full prompt is logged in the JSON — read it before
concluding the system is broken.

### 8. (Later) Decay

After your summaries have aged, compress the old ones:

```bash
python run.py decay --older-than 30 --target-words 50
```

**What happens:** every Layer 2 entry created more than 30 days ago is
shortened to ~50 words. The longer predecessor is moved to
`memory_data/layer2_archive/<date>/`.

```bash
python run.py archive --day 2026-04-26
```

Lists the versions for that day, with reasons (`compressed`,
`rebuilt_from_layer1`, `replaced_by_restore`) and word counts.

```bash
python run.py restore --day 2026-04-26 \
    --archive memory_data/layer2_archive/2026-04-26/v1_<timestamp>.json
```

Brings v1 back as the current Layer 2 entry. The thing it replaces
(say, v2) is itself archived first, so nothing is overwritten without
a record.

---

## What "pass" looks like

For Experiment 003 to count as passing all three pre-registered criteria:

1. `pre-flight` exits zero with `0/20 recognised`.
2. `canary` exits zero with `0/N leaked`.
3. At least one factual `query` returns a correct answer using only
   Layer 2 (no fallback).

When all three hold, you have an architecturally enforced result, not a
rhetorical claim. The two JSON files in `results/` are the evidence —
they contain every prompt and every response. Link them from
`tasks/chronos/scientific_log.md`.

If any test fails, that's still a result. Log the failure, deepen the
entity list or paraphrase more fingerprints, re-run. The pre-registered
log entry has space for an Incident block in the same style as
Incident 002-Alpha — failed-then-corrected runs are *more* credible than
silent successes.

---

## Troubleshooting

**`pre-flight` fails on a chunk that looks fine to you.**
The model's response (in the JSON) will name the cue. Common culprits:
brand names ("IKEA"), specific medical conditions, distinctive
sentence cadences ("I am Jack's…"). Add a regex for the cadence to
`fingerprints.json`, paraphrase, re-run.

**Anonymizer reports zero fingerprint hits.**
Almost certainly your fingerprint list is too thin, not that the text
is clean. Famous texts always have fingerprints; if the regex misses
them, the LLM won't.

**`canary` flags a leak with no source-specific terms.**
You probably hit the heuristic for confident identity-bridging
("they are the same"). Read the full response in the JSON. If the model
confidently identified pseudonym ↔ real-name based on hand-wavy reasoning
about names alone, that's a true leak. If it was something else
("they are the same in that both are characters in the passage"), it's
a false positive worth refining the heuristic for.

**Ingest produces only one chunk from a long file.**
The chunker splits on blank lines (`\n\n`). If your file uses single
newlines between paragraphs, fix it: `awk '{print; print ""}'` or open
in an editor and double-space.

**`memory.py` somehow grew a `vocab_store` import.**
Roll it back. The architecture is broken. Read the trust-boundary
section at the top of `memory.py` again.

**Ollama times out on `summarise`.**
The chunk dump may be too large. Either reduce the corpus to a chapter
or two for validation, or split it across multiple "days" by passing
explicit `--day` values. The Control Plane vs Data Plane benchmark
showed that long contexts in thinking mode can saturate the KV cache
into a 20-minute timeout. Use `think=False` (the default in
`ollama_client.py`) for summarisation.

---

## Limitations to declare in the writeup

- **Hand-curated entity list.** NER would scale this, at the cost of
  precision. For a published validation, manual curation is more
  defensible.
- **Manual fingerprint paraphrase.** Has to be — automating it via the
  LLM voids the test.
- **Layer 1 fallback uses concatenation, not embedding retrieval.**
  Adequate for one book; insufficient for production-scale corpora.
  Obvious next step: FAISS or sqlite-vss over Layer 1.
- **Vault is `chmod 600` only.** No encryption at rest. Acceptable for
  this experiment; not acceptable for real use. For real use, `age` or
  `sops` the file, and store the key separately.
- **One LLM provider.** All tests run against Ollama-compatible local
  models. The privacy claim doesn't depend on this; the architecture
  works against any HTTP LLM endpoint.

---

**Companion essay:** [Every Company Can Be a Palantir Now](https://localfirstai.eu/posts/every-company-can-be-a-palantir-now/)

**Pre-registered log entry:** `tasks/chronos/scientific_log.md` →
Experiment 003.
