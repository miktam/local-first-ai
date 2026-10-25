# exp_054: what leaves the machine during an Exposure Review

Aprimerose's Exposure Review tells each client, in writing, what leaves the machine during a review. This experiment watched a review of a fictional subject and checked those written claims one by one. Registered and run on 2026-10-25.

**Result.** Both controls passed, so the recorder demonstrably sees DNS, HTTP and spawned lookups. Four of the five claims hold:
- the review's other commands make no network call;
- every model call stays on the machine;
- only IANA and the domain registries are contacted;
- the client's DNS resolver never sees a domain from their messages;
- offline sends nothing.

One failed: X's and LinkedIn's link shorteners (`t.co`, `lnkd.in`) were sent to registries, although the documents say platforms are never looked up. Parry was fixed the same day, and the audit, re-run unchanged, passes all five (`runs/20261025T153728Z`). Details: [RESULT.md](RESULT.md).

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration: the claims under test, method, controls, H1–H5 |
| [RESULT.md](RESULT.md) | verdicts, what left per message, what changes |
| `kit/` | the audit hook (`hook/sitecustomize.py`), the corpus generator, the controls, the runner and the scorer, as run |
| `data/messages/` | the 15 test messages: invented domains, generic text, no lures |
| `runs/20261025T093602Z/` | every step's audit log (`logs/*.jsonl`), `lsof` samples, outputs, the registry caches, `score.json` |

**Not here.** The Exposure Review code (`exposure.py`) and Parry, the message check it calls, live in private repositories. The kit calls them by path. The audit logs show exactly which programs, hosts and queries they used.
