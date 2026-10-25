# exp_055 result: Exposure Review's message check catches new-domain lures and little else; the local model alone does better

**Run:** `runs/20261025T095614Z`, 2026-10-25, on the Mac mini (P, R, G) and in blind Opus 5.5 subagents (O). 220 messages: 110 lures and 110 genuine. Every arm answered all 220, except G, which returned 22 empty answers (below). These count as not flagged, as registered. The scorer applied Amendment 1 (the booking page's two ids count as one).

## Verdicts

| | Claim | Result | Verdict |
|---|---|---|---|
| H1 | P catches ≥ 80 % of lures and flags ≤ 15 % of genuine messages | catches 83/110 = **75.5 %** (95 % CI 67–83 %); flags 14/110 = 12.7 % | **fail** (catch) |
| H2 | P names an exploited item on ≥ 70 % of lures | 56/104 = **53.8 %** (mean recall 0.41) | **fail** |
| H3 | P's balanced accuracy ≥ G's (the glue over the local model alone) | P 0.814, **G 0.882** | **fail** |
| H4 | O − P ≤ 0.10 (the cost of staying local) | O 0.950 − P 0.814 = **0.136** | **fail** |
| H5 | Parry's model step catches more at ≤ 3 extra false alarms | it catches 15 more lures (p < 0.001) but adds **14** false alarms | **fail** |

## The numbers

| Arm | Lures caught | Genuine flagged | Balanced accuracy | Item named |
|---|---|---|---|---|
| R: Parry's rules only | 68/110 (62 %) | **0**/110 | 0.809 | 54 % |
| P: Exposure Review as built | 83/110 (75 %) | 14/110 (13 %) | 0.814 | 54 % |
| G: gemma4:26b alone, local | 92/110 (84 %) | 8/110 (7 %) | 0.882 | 85 % |
| O: Opus alone | 105/110 (95 %) | 6/110 (5 %) | 0.950 | 100 % |

No arm put a genuine message at likely-scam except G and O, once each.

## Where the message check is blind

Lures caught, by the age of the lure's domains:

| | New domain (72) | Older, aged or compromised (24) | No domain: chat with no link (14) |
|---|---|---|---|
| R rules | 64 | 3 | 1 |
| P as built | **72** | 10 | **1** |
| G gemma alone | 66 | 14 | 12 |
| O Opus | 72 | 19 | 14 |

**Parry is a domain-age detector.**
- It catches every lure on a new domain.
- It misses most lures on an aged or compromised domain (10 of 24).
- It misses almost every link-free social-engineering message, such as "it's me on a new number", a call-me request or a one-time code (1 of 14).

Those last two kinds are what a careful attacker uses against a high-value person. The fake-recruiter lure in Andrei's own campaign is that kind. The local model alone catches 12 of the 14 link-free lures; Opus catches all 14.

By the lure's ask, Parry is weakest on code requests (5 of 10), reply-baits (9 of 14) and call requests (6 of 9).

## Where the false alarms come from

Parry's rules raised **no** false alarm. All 14 of P's come from its one local model step: 10 on genuine messages from established domains, 4 on genuine young domains. That step adds 15 catches and 14 false alarms, nearly a wash, because it sees only part of the picture and can only add points.

## Attribution ("why you were targeted")

The join names an exploited footprint item on 54 % of lures, finding on average 41 % of what each lure used. It matches exact strings, and lures paraphrase: "your Harborlight talk" against the talk's title. Given the footprint and the message, the local model names one on 85 % (recall 0.81), and Opus on all of them.

## G's empty answers

22 of G's 220 answers (10 %) were empty: it thought until it hit the 8,192-token context. Under the registered rule they count as not flagged, so G's catch rate is understated and its false-alarm rate slightly flattered. This is the same failure as the bot's in exp_052 and exp_053.

## SO WHAT

- **The product claim.** The message check as built would miss about 1 lure in 4 overall, and nearly all of the kind that matters most for Exposure Review's clients: aged-domain and link-free social engineering. Its precision is excellent (the rules never cried wolf), so what it does flag is right. **Nothing on the page or in the paperwork should claim detection coverage.** The record should say plainly that the check is strongest against new domains, and that what it doesn't catch is not cleared.
- **The architecture.** Here the local model given the whole picture (footprint, registry facts, message) beat the deterministic pipeline (0.882 against 0.814), at 34 s a message against 2 s. That runs against our usual result that glue beats a lone model. The reason is visible: Parry's model step is a narrow leaf that sees part of the message and can only raise. The better shape is to keep the rules as hard evidence that cannot be overruled, and to give the local model the whole picture as the primary reader. That design is the next experiment. Its first fix is the empty answers: a larger context or a thinking cap.
- **The cost of staying local.** Opus is 0.068 ahead of the local model alone, and 0.136 ahead of the pipeline as built. The gap is mostly aged-domain lures (19 against 14 caught).
- **B, the self-fixing rules loop.** It now has its target list:
  - link-free asks: new number, call-back, one-time code;
  - aged-domain impersonation of known correspondents and suppliers;
  - paraphrased hooks.
- **Andrei's 7 real incidents** stay the regression floor for any change.
