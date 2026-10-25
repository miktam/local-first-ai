# exp_054: what leaves the machine during an Exposure Review

**Registered 2026-10-25, before any measured run.** Andrei approved it ("ok, go", after choosing it as experiment C of three for Exposure Review). Kit: `kit/`. The corpus is frozen by the hashes at the end.

## Why

Aprimerose's Exposure Review tells a client, in writing, what leaves the machine during a review. The facts behind those documents are in `tasks/exposure-report/paperwork/WHAT-THE-CODE-DOES.md` §1. They were checked by reading code, never by watching a review run. The delivery gate holds those claims as unverified until the first session (December 2026). This experiment watches a review run and checks each written claim against what happened.

## The claims under test (the annex's own words, condensed)

1. **Inference is local.** No model call leaves the machine; the leaf talks to Ollama on `127.0.0.1:11434`.
2. **The rest of a review makes no network call at all:** capture, inventory, assessment, report, day-30 delta. The one outbound flow is the message check.
3. **Up to twelve domains per message are disclosed** (`AGE_CAP = 12`). Excluded and never looked up: platforms, the subject's own domains, registries, known correspondents and free-hosting domains.
4. **At the `registry` tier**, the only contacts are:
   - IANA's RDAP bootstrap file (`data.iana.org`, cached);
   - the domain's registry RDAP service;
   - when a TLD has no RDAP, `whois.iana.org`, which is sent the TLD alone, then the registry's own whois server.

   No SPF, `_dmarc`, MX or name-server records are read. The resolver sees the hostnames contacted, which are registry hostnames.

## Method

- **Subject.** A fresh copy of the fictional demo subject (`tasks/exposure-report/demo`, "Iris", on RFC 2606 reserved names), with every command run as a client run would.
- **Recording.** An audit hook (`kit/hook/sitecustomize.py`, loaded through `PYTHONPATH`, so child Pythons record too) logs, per step:
  - every socket connect, every name resolution with the addresses returned, every URL request and HTTP connection;
  - every spawned program with its full command line.
  
  `whois` and `dig` run as spawned programs, so their target server and query are on that command line.
- **Cross-check.** `lsof` samples the step's open internet sockets every 0.2 s, descriptively.
- **Fresh cache.** Parry's lookup cache (`TRIAGE_CACHE`) is a fresh folder per tier, so every lookup the code would make is made.
- **Corpus.** 15 messages (`kit/corpus54.py`, `data/messages/`): 14 e-mails and one pasted chat message, generic text, no crafted lures. Every domain is invented except the platforms in M07, M13 and M14. They cover:
  - RDAP TLDs (.com .net .pl .fr .xyz .top .online .info) and TLDs without RDAP (.es .de .io .it .ru .se .ch), so the whois path runs;
  - a lookalike, a link whose text shows the subject's own domain, a redirect parameter, platform shorteners;
  - a message with 21 domains, to test the cap of 12;
  - a message whose domains are all excluded classes (platforms, free hosting, the subject's own);
  - Authentication-Results headers;
  - a repeat of an already looked-up domain.
- **Steps, in order:**
  1. control C2;
  2. `build`, `snapshot`, `delta`, `week` and `report` on the subject;
  3. `triage` of each message at the default tier, with the model leaf;
  4. the same with `--offline`;
  5. control C1.

## Controls (if either fails, the audit is void)

- **C2.** A script resolves `example.com`, fetches `http://example.com/` and spawns `dig example.com`. The log must show all three.
- **C1.** Parry's `check.py --net full` on M01. The log must show `dig` for the sender's mail records, the DNS reading that the `registry` tier claims never to do.

## Hypotheses

| | Claim | Passes if |
|---|---|---|
| **H1** | The review's other commands make no network call | `build`, `snapshot`, `delta`, `week`, `report`: no resolution, no connection, no URL request, no network program, no model call |
| **H2** | Every model call stays on this machine | at least one model call is seen, and every connection or resolution on port 11434 is loopback |
| **H3** | The registry tier contacts only registries | in every message check, every name resolved and every URL is `data.iana.org` or an RDAP host named in IANA's bootstrap file. Every `whois` goes to `whois.iana.org` or to a server IANA named for that TLD. No `dig` runs. The resolver is never asked about any domain in the message |
| **H4** | Disclosure is capped and exclusions hold | in every message check, the domains sent to a registry (RDAP paths and whois queries) number at most 12, are all from that message, and include none it should exclude; `whois.iana.org` is sent bare TLDs only |
| **H5** | Offline sends nothing | with `--offline`, nothing beyond the loopback model call: no resolution, connection, URL or network program |

**Descriptive only.**
- Domains disclosed per message, against the domains it carries.
- The shorteners (M13) and the Authentication-Results message (M08).
- The repeat message (M15), which should cause no second lookup.
- How often the bootstrap is fetched.
- Endpoints seen by `lsof` that the logs do not explain.
- Exit codes.

## Limits

- **What the hook sees.** It covers Python and the programs Python spawns. A program's own network use (here `whois`) is seen through its command line and through `lsof`, not packet by packet. A capture by `tcpdump` would need root, which this account does not hold for that tool.
- **The tier.** Only the review's default tier, `registry`, is tested. `full` is exercised only as control C1.
- **Not tested here.** The container, the cache's placement inside it, and deletion, which are claims about storage.
- **The subject.** It is fictional, and its correspondents list is empty, so "known correspondents" is not exercised.

## Frozen inputs (sha256, first 12)

M01 3f1f205d47dd · M02 aac4446f51c6 · M03 7ecb642cda6c · M04 484238086d05 · M05 36dfaae47dee · M06 6ac09cd7ed07 · M07 9178e3faaea6 · M08 a013c6390cc3 · M09 bbd3cd3bd462 · M10 5c9cf085bc15 · M11 cefa3d954c9d · M12 a5f6711fc0c9 · M13 570dfbcf795f · M14 72c4ea5fabd9 · M15 5e1d6d2f5422 · MANIFEST 42b5daec2106

**Dev before registration.** Control C2 and one offline, model-off triage of a dummy message (not in the corpus) ran with the hook, to check that it records. The triage spawned only `git rev-parse` and `git status` on Parry's own folder, for its version stamp.
