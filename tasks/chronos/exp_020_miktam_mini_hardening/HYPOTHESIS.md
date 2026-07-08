# Experiment 020 — Hardening and Red-Teaming the Inference Node

*Pre-registered: 2026-07-08 · Status: pre-registered, formal execution pending*

**Pre-registration:** [`tasks/chronos/scientific_log.md`](../scientific_log.md) → Experiment 020

---

## What this experiment tests

The local-first architecture's core claim — client documents never leave the operator's hardware — is only as strong as that hardware's own security posture. This experiment tests, empirically rather than by config inspection alone, whether the machine that runs local inference (miktam-mini) actually holds up: can an unprivileged local process escalate privilege or read protected data via existing sudoers grants, and is the local model's API actually reachable without authentication from elsewhere on the network.

An informal recon pass (Phase 0 of the hardening task, not part of this experiment's formal evidence) read the machine's `sudo -l -n` output and initially misread two entries — `/bin/cp` and `pmset` — as unrestricted NOPASSWD grants. Direct invocation testing during that same recon disproved this: `sudo -n cp` and `sudo -n pmset -g` both still prompted for a password despite appearing broad in the static listing (most likely a line-wrap artifact in how the listing rendered, conflating two adjacent rules). That correction is itself load-bearing for this experiment: it is why the hypotheses below are framed as testable claims to verify by direct invocation, not by re-reading `sudo -l` output and trusting it.

---

## Hypothesis

**Primary (H1):** `sudo -n /usr/bin/tail <arbitrary-root-only-readable-file>` succeeds without a password prompt and returns real file content, for at least one file not otherwise readable by the `miktam02` user — confirming unrestricted root-level file read via an unscoped NOPASSWD grant.

**Secondary (H2):** Before any fix, Ollama's HTTP API on port 11434 is reachable and fully functional (a real inference request succeeds) from at least one other device already on the Tailscale network, with zero credentials presented. After adding an authentication layer, the identical unauthenticated request is rejected, while a request presenting the correct credential still succeeds.

**Tertiary (H3):** Of the remaining sudoers NOPASSWD grants (`shutdown`, `lsof`, `asitop`, `powermetrics`/`killall powermetrics`, the `tailscaled`/`ollama` launchctl bootstrap/bootout pairs, the `pharos-deploy` script), each is invoked directly to confirm which are genuinely passwordless (not just apparently so in `sudo -l`), and none beyond `tail` grants unrestricted file read or write — i.e. each remaining grant is either scoped to a fixed command/path, or grants a narrow, low-severity capability (process visibility, telemetry, service restart, availability) rather than confidentiality-breaking file access.

**Null hypothesis:** None of H1–H3 hold in practice — either the sudoers grants are more narrowly scoped than they appear, or some other control (e.g. SIP, entitlements) blocks the read/write despite sudo's own gate passing.

---

## Test methodology

Each hypothesis gets one script under this directory, invoked directly (no config-file re-reading), writing a single JSON result to `results/` with an ISO-timestamped filename. Exit codes: `0` = hypothesis confirmed, `1` = hypothesis refuted, `2` = inconclusive/blocked, `3` = test itself errored.

- **H1** (`test_h1_tail.sh`): attempt `sudo -n tail` against a file only root can read; record success/failure and the actual bytes returned (redacted in the committed result if sensitive) as proof, not just an exit code.
- **H2** (`test_h2_ollama_exposure.sh`): from this machine, curl the Ollama API with no auth header, before and after the fix; separately, request the same test be run from another tailnet device for the "external" leg (documented, not scripted, since it requires a second machine).
- **H3** (`test_h3_remaining_sudoers.sh`): invoke each remaining NOPASSWD-listed command with `-n` and a harmless/read-only argument (never an actual shutdown, never a real service bounce mid-test), recording pass/fail per command.

Nothing in any script writes to, modifies, or deletes a real system file, sudoers entry, or launchd job. `results/` and this file are append-only once a run completes — a corrected re-run gets a new timestamped file, not an edit to the old one.
