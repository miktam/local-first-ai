# exp_056: can a hostile message subvert its own triage?

Exposure Review's message check (Parry) reads hostile messages by design. Before any channel feeds it messages that no human chose, this experiment asks whether a message's own text can lower its verdict, silence the model step, push its evidence out of view, or reach the screen as itself. Registered 2026-10-25.

**Status: run 2026-11-10; the gate is not passed.** With the six frozen texts added to the end of 24 messages (240 runs), H1 and H2 do not hold: 27 of 144 variants came out with a lower verdict, and on 19 of 72 lure variants the model step's points went to zero. H5, C2 and C3 hold: with the model unreachable, every verdict fell back to the rules and said so. The neutral control N1 (120 runs) failed its own C2 on one base, so its H1 and H2 are void (3 of 24, 1 of 12). The six texts stay held until a fix to Parry has passed this gate again. Details: [RESULT.md](RESULT.md). **A fix was opened the same day; the same gate is registered to run again in Amendment 4, with a false-alarm rule beside it. That re-run has not been reported yet.**

How it got here:
- **Amendment 1** froze the six class-A instruction-injection payloads by sha256 before any kit existed.
- **Amendment 2** narrowed the first run to classes A and F (H1 over class A, H2, H5). H3 and H4 are deferred.
- While the attack material and the variant builder were being written, the authoring session was stopped three times by a safety classifier. The builder was then written outside that session, by Andrei with a local model.
- **Amendment 3** (2026-11-10, before the run) pinned the kit, the model and the code, and added the neutral control.

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration (properties, variant classes, H1–H5, controls), Amendments 1 and 2, the parked status of 2026-10-25, Amendment 3 (the kit as it ran, a neutral control, what is pinned), and Amendment 4 (the fix to the message check, the same gate registered again, and the measure of what the fix costs) |
| [RESULT.md](RESULT.md) | the gate, the verdicts, the neutral control, how the violations split, the attribution, what ran, what was not measured, what changes |
| [payloads/README.md](payloads/README.md) | what each class-A payload attempts |
| [control/N1.txt](control/N1.txt) | the neutral control text, as hashed in Amendment 3 |
| `kit/` | `run56.py`, `score56.py`, `select56.py`, `build56.py`, `test_score56.py` and `README.md` as they ran; `verify_build56.py`, the byte check of the build; `attribute56.py`, written for the result |
| `runs/six/`, `runs/n1/` | each run's `results.jsonl` and `meta.json` as written; `audit/`, the audit logs with long strings cut to 160 characters (RESULT.md says why); `score.json` and `score.txt`, the scorer's output |
| `runs/verify-output-build1.txt`, `runs/verify-output-build2.txt` | the byte check's output on the first build (failed, not run) and the second (passed, run) |

**Not here.** The six payload files are held back: Amendment 3 publishes them with the result only if H1, H2 and H5 all hold, and H1 and H2 did not. Their hashes are in Amendment 1, so a later copy can be checked against what was frozen. The variant files, which contain them, are not published either, nor each run's `cache/` and `subject/`. The 24 bases are exp_055's published messages of the same ids; their hashes are in `meta.json`. Parry and Exposure Review live in private repositories, as for exp_054 and exp_055.
