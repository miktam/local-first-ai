# exp_055: does Exposure Review's message check catch lures built from a person's own footprint?

A benchmark of 110 targeted lures and 110 genuine messages, written from a fictional person's public footprint, with labels by construction. It was answered by four arms:
- Exposure Review's message check as built;
- its rules alone;
- gemma4:26b alone, locally;
- Opus alone.

Registered and run on 2026-10-25.

**Result.** All five hypotheses fail:
- **As built:** the check catches 75 % of lures and wrongly flags 13 % of genuine messages. Its rules alone never flag a genuine message, but they catch only 62 %.
- **Where it is blind:** it is close to a domain-age detector. It catches every new-domain lure, 10 of 24 on aged or compromised domains, and 1 of 14 link-free ones.
- **The other arms:** gemma4:26b alone, given the whole footprint, catches 84 % with 7 % false alarms. Opus catches 95 % with 5 %.

Details: [RESULT.md](RESULT.md).

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration and Amendment 1 |
| [RESULT.md](RESULT.md) | verdicts, numbers, where the check is blind, what changes |
| `data/FOOTPRINT.txt`, `data/WRITER_SPEC.md` | the fictional subject's 26 footprint items, and the writers' specification |
| `data/raw/` | the four writers' records, with the construction labels |
| `data/messages/`, `data/MANIFEST.json` | the 220 messages as the arms saw them (opaque ids X001–X220), and the key |
| `data/cache/` | the registry facts, one creation date per domain, served offline |
| `kit/` | builder, arms, Opus batch builder, prompt, scorer, as run |
| `runs/20261025T095614Z/` | every arm's answers (`P`, `R`, `G`: `results.jsonl`; `O`: batches and verdicts), `score.json` |

**Everything here is fictional:** the person, every organisation and domain, drama-range phone numbers, a test IBAN. No lure is operational.

**Not here.** The Exposure Review code and Parry are private; the kit calls them by path. The per-message working folders of arms P and R are omitted: they are a copy of the demo subject and its run folders, and their results are in `results.jsonl`.

**The published copy differs from the run in five domain names.** Before publishing, every invented domain was checked against DNS. Five turned out to be registered by real parties: `brightpost.online`, `brindlewick.co.uk`, `mailhaven.co`, `postlark.net` and `vaultgrove.com`. Seven genuine messages gave invented people mailbox addresses at them. In every published file they are replaced by reserved names (`brightpost.example` and so on), so no address here can belong to anyone.

The run used the originals. Their sha256 is in [HYPOTHESIS.md](HYPOTHESIS.md) (Frozen inputs): MANIFEST `1658078c…`, cache `8bba0092…`, messages `476d8bd9…`. All seven messages are genuine. Arms O and G saw the original names, so a re-run on this copy may differ slightly on those seven.

