# exp_055: does Exposure Review's message check catch lures built from a person's own footprint?

**Registered 2026-10-25, before any measured run.** Andrei approved it ("ok, go", as experiment A of three for Exposure Review). Kit: `kit/`. The corpus is frozen by the hashes at the end.

## Why

Exposure Review's message check (Parry, joined to the person's footprint inventory) has been validated only on Andrei's 7 real incidents of September and October 2026, which its own record calls "a regression floor, not accuracy". This experiment gives it a first accuracy number. It also measures what keeping it local costs, against the same task given to a frontier model.

## The subject and the corpus

- **The subject.** The fictional demo subject, Iris Kestrelmoor (`tasks/exposure-report/demo`, on reserved names), with her inventory of 26 footprint items: 13 identity, 8 artefacts, 5 channels (`data/FOOTPRINT.txt`).
- **The corpus.** Four Fable 5.1 writers worked to one specification (`data/WRITER_SPEC.md`):
  - **lures**, each built from named footprint items: talk and podcast follow-ups, analysts' institute renewals, supplier payment changes, trustee and colleague impersonation, registry-number notices, sailing club, board-seat headhunting, "new number" chats, booking-page abuse, one-time codes, alumni;
  - **genuine messages** using the same facts legitimately, a share of them alarming in isolation.
  
  Everything is fictional: invented organisations, invented domains on real TLDs, drama-range phone numbers, no operational content.
- **Labels are by construction.** A lure is a lure because it was written as one. **No model filters the set**, so no arm is scored on messages its own model family selected. Checks are by code only (`kit/build55.py`).
- **Blind ids.** Every message gets an opaque id (X001…) in a fixed shuffled order (seed 55), so no arm sees its kind.
- **Registry facts.** Every domain's creation date, as the writer set it, is served from a prepared lookup cache (`data/cache`). The arms run `--offline`: **nothing leaves the machine** except arm O, which goes to Opus as described below.

## Arms

| Arm | What answers | Output |
|---|---|---|
| **P** | Exposure Review's message check as built: `exposure.py triage --offline` (Parry's deterministic rules, the registry facts, its one local model step on gemma4:26b, and the footprint join) | band; the join's hooks (footprint items, strong or weak) |
| **R** | The same with `--no-llm`: rules and registry facts only | as P |
| **G** | gemma4:26b alone, given the footprint, the registry facts and the message (`kit/prompt55.md`); temperature 0, seed 55, Ollama's default thinking | band; the footprint items it names |
| **O** | Opus 5.5, blind subagents, the same prompt as G, in batches of 28 | as G |

The bands are `benign`, `suspicious` and `likely-scam`. A message counts as **flagged** if it is `suspicious` or `likely-scam`.

## Measures

- **Catch rate:** lures flagged. **False-alarm rate:** genuine messages flagged, also reported as `likely-scam` alone.
- **Balanced accuracy:** (catch rate + 1 − false-alarm rate) / 2.
- **Attribution.** The footprint items a lure declares, apart from her name and the channel it arrived on, are its distinctive items. A lure counts as attributed when the arm names at least one of them: the join's hooks for P and R, `used_items` for G and O. Mean recall is reported too.
- **Intervals.** Wilson 95 % intervals for every rate.

## Hypotheses

| | Claim | Passes if |
|---|---|---|
| **H1** | The message check catches targeted lures without crying wolf | P catches ≥ 80 % of lures **and** flags ≤ 15 % of genuine messages |
| **H2** | It tells the person why they were targeted | P names at least one distinctive exploited item on ≥ 70 % of the lures that have one |
| **H3** | The glue earns its place over the local model alone | P's balanced accuracy ≥ G's |
| **H4** | Keeping it local costs little | O's balanced accuracy − P's ≤ 0.10 |
| **H5** | Parry's local model step earns its place | P catches lures R misses (one-sided McNemar p < 0.05) at no more than 3 extra false alarms |

**Descriptive only.**
- Catch rate by tactic, by ask, and for lures on new domains, older (aged or compromised) domains, or no domain.
- False alarms on genuine messages with a young domain.
- Results by channel.
- O against G: frontier alone against local alone.
- Unanswered or unparsed outputs, counted as not flagged.

## Limits

- **Synthetic, and one writer family.** The messages are written by one model family. Real campaigns differ, and the 7 real incidents remain the regression floor.
- **Registry facts are the writers'.** The domain ages are chosen by the writers, which tests the logic, not live registry data. exp_054 tested the live path.
- **One pass.** One pass per arm. P's model step and G run at temperature 0; O is one sample.
- **Thresholds.** The thresholds are judgment calls, set before any message was scored.

## Frozen inputs

| Input | sha256 |
|---|---|
| `data/MANIFEST.json` | `1658078c55afcf5d1f392b6370d20372acfd680cf262164a4deb04612e2bdb29` |
| `data/cache/lookups.json` (147 seeded domains) | `8bba0092420208bf780db3a23c47e60e99186d808b59fbe97ef925fc262c3080` |
| `data/messages/*.eml` then `*.txt`, concatenated | `476d8bd9cfc6f72912722918f810b5fc1e32b6e8cd35f3e1453e116caa145e05` |

**The corpus.**
- 220 messages: 110 lures (LA01–LA55, LB01–LB55) and 110 genuine (BA01–BA55, BB01–BB55); 182 e-mails, 25 WhatsApp, 13 SMS. All passed the code checks.
- **One conflict.** Two lures use the same lookalike domain, `vennholt-partner.com`, with different creation dates. One domain has one age, so the first record's (2026-10-19) applies to both, and LB29 counts as a new-domain lure.

**Dev before registration.** Arms P and G ran once on a dummy message that is not in the corpus, to check the kit:
- P flagged it suspicious (a 5-day-old domain) and named no hook, although it mentions the Harborlight talk.
- G flagged it suspicious, named the talk, and took 34 s.

No corpus message was scored before registration.

## Amendment 1 (2026-10-25, before any arm was scored)

Her booking page is in the footprint twice, as an artefact (`A-f3776b`) and as a channel (`C-69db50`), with the same URL. Attribution leaves channel items out, so a lure that draws on the booking page would count as unattributed when an arm cites the channel form. For attribution, naming either id now counts as naming `A-f3776b`, for declared and reported items alike. Raised by an arm-O agent's note about how it cited the page, before any result was read; nothing else changes.
