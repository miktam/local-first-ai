# exp_056: can a hostile message subvert its own triage?

Exposure Review's message check (Parry) reads hostile messages by design. Before any channel feeds it messages that no human chose, this experiment asks whether a message's own text can lower its verdict, silence the model step, push its evidence out of view, or reach the screen as itself. Registered 2026-10-25.

**Status: run 2026-11-10, twice.** The first run did not pass the gate: with the six frozen texts added to the end of 24 messages (240 runs), 27 of 144 variants came out with a lower verdict, and on 19 of 72 lure variants the model step's points went to zero ([RESULT.md](RESULT.md)). A fix to the message check was pinned in Amendment 4 the same day, and the same gate was run again that evening: **the re-run passed.** H1 0 of 144, H2 0 of 72, H5, C2 and C3 hold, and the neutral control N1 holds too. On exp_055's 220 messages the fixed check flagged 16 of 110 genuine messages (14 before; the registered limit is 19) and caught the same 83 of 110 lures, at about five model calls and 3.2 times the seconds per message ([RESULT-RERUN.md](RESULT-RERUN.md)). The fix meets its registered rules; adopting it is Andrei's decision. The six texts are now published. A pass clears H1 (class A), H2 and H5 only: H3 and H4 (classes B–E) are unmeasured, so the mail sentry stays blocked.

How it got here:
- **Amendment 1** froze the six class-A instruction-injection payloads by sha256 before any kit existed.
- **Amendment 2** narrowed the first run to classes A and F (H1 over class A, H2, H5). H3 and H4 are deferred.
- While the attack material and the variant builder were being written, the authoring session was stopped three times by a safety classifier. The builder was then written outside that session, by Andrei with a local model.
- **Amendment 3** (2026-11-10, before the run) pinned the kit, the model and the code, and added the neutral control.
- **Amendment 4** (2026-11-10, before the re-run) pinned the fix to the message check, the same gate again, and a false-alarm rule on exp_055's messages.

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration (properties, variant classes, H1–H5, controls), Amendments 1 and 2, the parked status of 2026-10-25, Amendment 3 (the kit as it ran, a neutral control, what is pinned), and Amendment 4 (the fix to the message check, the same gate registered again, and the measure of what the fix costs) |
| [RESULT.md](RESULT.md) | the first run: the gate (not passed), the verdicts, the neutral control, how the violations split, the attribution, what ran, what was not measured, what changes |
| [RESULT-RERUN.md](RESULT-RERUN.md) | the re-run with the fix: the gate (passed), the verdicts beside the first run's, the neutral control, the false-alarm rule and the cost on exp_055's messages, what the model step was asked, what ran, what was not measured, what changes |
| [payloads/](payloads/) | `A1.txt` to `A6.txt`, the six class-A texts, published with the re-run's result (hashes in Amendment 1); [payloads/README.md](payloads/README.md), what each attempts |
| [control/N1.txt](control/N1.txt) | the neutral control text, as hashed in Amendment 3 |
| `kit/` | `run56.py`, `score56.py`, `select56.py`, `build56.py`, `test_score56.py` and `README.md` as they ran; `verify_build56.py`, the byte check of the build; `attribute56.py`, written for the first result; `compare56.py` and `views56.py`, written for the re-run's result |
| `runs/six/`, `runs/n1/` | the first run: each run's `results.jsonl` and `meta.json` as written; `audit/`, the audit logs with long strings cut to 160 characters (RESULT.md says why); `score.json` and `score.txt`, the scorer's output |
| `runs/six-r2/`, `runs/n1-r2/` | the re-run, in the same form |
| `runs/exp055-views/` | exp_055's R and P arms run again with the fixed check: `results.jsonl` of each and exp_055's `score.json`, as written |
| `runs/views-output.txt` | the output of `kit/views56.py`: what the model step was asked per message in the re-run, read from the run folders that are not published |
| `runs/verify-output-build1.txt`, `runs/verify-output-build2.txt` | the byte check's output on the first build (failed, not run) and the second (passed, run) |

**Not here.** The variant files, which are each base, a blank line and a text; each run's `cache/` and `subject/` (the check's working files, which hold the messages). The 24 bases are exp_055's published messages of the same ids; their hashes are in `meta.json`. Parry and Exposure Review live in private repositories, as for exp_054 and exp_055.
