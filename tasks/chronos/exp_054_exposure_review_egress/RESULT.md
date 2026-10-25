# exp_054 result: four of the five written egress claims hold; platform link shorteners leave the machine

**Run:** `runs/20261025T093602Z`, 2026-10-25, about 1 minute, on the Mac mini. Both controls passed, so the audit is valid:
- **C2:** the recorder saw `example.com` resolved, fetched and `dig`ged.
- **C1:** Parry at `--net full` spawned `dig` for the sender's NS, TXT and mail records, which the registry tier must not do.

`lsof` saw no internet endpoint the logs do not explain. One run slip: the console log file failed to open (its folder did not exist yet), so the final exit code 1 belongs to `tee`. Every step's own logs are complete.

## Verdicts

| | Claim (annex §1) | Result | Verdict |
|---|---|---|---|
| H1 | Capture, inventory, assessment, report and delta make no network call | `build`, `snapshot`, `delta`, `week`, `report`: no resolution, connection, URL, network program or model call | **pass** |
| H2 | Inference is local | 120 model connections, every one to `127.0.0.1:11434` | **pass** |
| H3 | The registry tier contacts only IANA and registries; no DNS records | every resolution and URL was `data.iana.org` or one of 379 RDAP hosts in IANA's bootstrap file; every `whois` went to `whois.iana.org` or one of the 12 servers IANA named; no `dig`; **the resolver was never asked about any domain in a message** | **pass** |
| H4 | At most 12 domains per message, never platforms, own, registries or free hosting | the cap held (the 21-domain message disclosed 10), and the subject's own domains, platforms and free hosting were never sent. **But in M13 the platform shorteners `t.co` (X) and `lnkd.in` (LinkedIn) were sent to registries** | **fail** |
| H5 | Offline sends nothing | no resolution, connection, URL or network program beyond the loopback model call | **pass** |

## What leaves, per message (descriptive)

| Message | Domains in it | Sent to a registry | Note |
|---|---|---|---|
| M01 .com sender | 1 | 1 | IANA bootstrap fetched once here, cached after; IANA's whois asked "com" only |
| M02 .es, M03 .de | 1 each | 0 | the registry gives whois only to registered IP addresses and runs no RDAP; Parry knows this and asks nothing |
| M05 own-domain link text | 2 | 1 | the subject's own domain was not sent; the link's real target was |
| M06 21 domains | 21 | 10 | under the cap of 12; the rest were on TLDs Parry does not query |
| M07 platforms, free hosting, own | 6 | **0** | every exclusion held |
| M08 Authentication-Results | 1 | 1 | |
| M13 shorteners | 3 | **3, including t.co and lnkd.in** | the H4 failure |
| M15 repeat of M01's domain | 1 | 1 | the domain was sent again: the cache did not spare a second lookup (descriptive) |

`whois.iana.org` was only ever sent bare TLDs (com, xyz, net, info, online, top, io, pl, fr, it, ru, se, ch, co).

## Why the shorteners leave

Parry excludes a domain from lookups when it is trusted, meaning a platform, the subject's own, a registry, or a free-hosting suffix. `t.co` and `lnkd.in` are in its shortener list (`SHORTENERS_OFF_PLATFORM`), not its platform list, so they are aged like any other domain. Their age says nothing about the link, for the same reason a free-hosting suffix's age says nothing. And the client was told platforms are never looked up.

## SO WHAT

- **For the delivery gate.** The egress claims can move from "unverified" to **measured**, except the exclusion list. Either:
  - Parry excludes platform shorteners from lookups (the better fix: their age is meaningless, as the code already argues for free hosting); or
  - the annex and the client documents say "platform link shorteners (t.co, lnkd.in) are looked up".

  Parry's code belongs to another session, so the code fix is for its owner, or for Andrei to approve.
- **For the product page.** It can say, with evidence: during a review, nothing leaves the machine but registry lookups for at most 12 domains per message the client brings. The model never leaves the machine, and the client's DNS resolver never sees the domains in their messages.
- **A regression check.** `kit/run54.py` plus `kit/score54.py` run in about a minute. Run before every Exposure Review release, they keep these claims true.
- **The M15 repeat** shows the lookup cache did not prevent a second disclosure of the same domain within a run. That is worth a look by Parry's owner (descriptive, no claim covers it).

## Addendum, 2026-10-25: the fix, and the audit re-run

At Andrei's go, Parry's `indicators.age_candidates` now skips platform link shorteners (`SHORTENERS_OFF_PLATFORM`: `t.co`, `lnkd.in`), as it already skipped free-hosting suffixes. A test pins it (`tests/test_indicator_fixes.py::test_platform_shorteners_never_aged`). Parry's suite passes 673/673, and Exposure Review's 599 (1 skipped).

The registered audit, re-run unchanged against the fixed code (`runs/20261025T153728Z`):
- **Controls:** both pass.
- **H1–H5:** all pass.
- **M13:** now discloses only its sender (`verbena-news.com`).
- **M07:** still discloses nothing.
- **M06:** still discloses 10 of 21.

The written claim, *platforms are never looked up*, is now true as measured. The verdict above stays as registered: the code under test failed H4.

