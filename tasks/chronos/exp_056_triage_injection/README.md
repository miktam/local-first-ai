# exp_056: can a hostile message subvert its own triage?

Exposure Review's message check (Parry) reads hostile messages by design. Before any channel feeds it messages that no human chose, this experiment asks whether a message's own text can lower its verdict, silence the model step, push its evidence out of view, or reach the screen as itself. Registered 2026-10-25.

**Status: registered, parked before any kit was built. Nothing has run.**
- **Amendment 1** froze the six class-A instruction-injection payloads by sha256 before any kit existed.
- **Amendment 2** narrowed the first run to classes A and F (H1 over class A, H2, H5). H3 and H4 are deferred.
- While the attack material and the variant builder were being written, the authoring session was stopped three times by a safety classifier. The build was parked there. To resume, the builder has to be written outside that session.

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration (properties, variant classes, H1–H5, controls), Amendments 1 and 2, and the parked status |
| [payloads/README.md](payloads/README.md) | what each class-A payload attempts |

**Not here yet.** The six payload files are held back until the run. Their hashes are in Amendment 1, so the published copy can be checked against what was frozen. They target a check that clients rely on, so they are published with the result, after the defence has been measured. Parry and Exposure Review live in private repositories, as for exp_054 and exp_055.
