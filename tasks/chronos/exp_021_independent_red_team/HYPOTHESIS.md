# Experiment 021 — Independent Red-Team Pass on the Inference Node

*Pre-registered: 2026-07-08 · Status: H3 complete (see `H3_FINDINGS.md`). H1 blocked until ~2026-07-10 — `miktam-mbp` is off the network for ~2 days. H2 ready to run any time, instructions below, needs the operator's interactive sudo.*

**Pre-registration:** [`tasks/chronos/scientific_log.md`](../scientific_log.md) → Experiment 021
**Builds on:** [`exp_020_miktam_mini_hardening/`](../exp_020_miktam_mini_hardening/)

---

## What this experiment tests

Exp_020's red-team phase was conducted by the same session, same account (`miktam02`), same machine that also built the fix it was testing. That's a legitimate first pass — it found and confirmed a real vulnerability — but it's weak as *independent* evidence: nothing in it rules out the tester having implicit context or access a genuinely separate party wouldn't have. This experiment re-tests the same class of claim from vantage points that don't share that context, plus one new angle exp_020 didn't cover: what's reachable through ordinary file permissions, with no privilege escalation needed at all.

---

## Hypothesis

**H1 (network, from a genuinely separate device):** From `miktam-mbp` — an already-authorized tailnet device, not miktam-mini testing its own interface — an unauthenticated request to the retired Ollama port (`11434` on the tailnet address) fails or times out; an unauthenticated request to the new proxy (`11435`) returns 401; the identical request with the correct bearer token returns 200. Reproduces exp_020's H2 from an actual independent machine rather than the same box checking itself.

**H2 (privilege boundary, from a genuinely separate account):** A freshly created, non-admin, no-special-grants local macOS account cannot invoke any of the `miktam02`-scoped NOPASSWD sudoers grants — in particular, cannot reproduce the exp_020 H1 finding (`sudo -n tail` reading `/etc/sudoers`). Confirms the grants are scoped to `miktam02` specifically, not something any standard account on this machine could stumble into.

**H3 (the more realistic threat model — no privilege needed at all):** Without sudo, without any escalation, what can a process already running *as* `miktam02` read? This is the account that runs Ollama, the document-processing pipeline, and every tool call in this automation context — if a compromised dependency or a crafted input ever achieved code execution in that context (no OS-level exploit required, just running as the account that already exists), what's sitting there for it to find? Specifically checked: SSH private keys and whether they're passphrase-protected, the Ollama proxy token, API keys exported in shell startup files or the current environment, and stored git credentials.

**Null hypothesis:** all three replicate exp_020's existing findings exactly with nothing new — meaning the internal-vantage pass in exp_020 was already sufficient and this experiment is purely confirmatory.

---

## Test methodology

Same conventions as exp_020: one script per hypothesis where independently executable, JSON evidence in `results/` with ISO timestamps, exit codes `0` confirmed / `1` refuted / `2` inconclusive / `3` blocked-on-external-dependency.

- **H1** requires either SSH access to `miktam-mbp` from this session, or the operator running three `curl` commands there directly and reporting the output back — this experiment cannot self-execute it. Status: blocked pending access.
- **H2** requires a throwaway standard account created on miktam-mini (`sudo sysadminctl -addUser ...`), a one-time step needing the operator's interactive password — same constraint exp_020 hit repeatedly. Status: blocked pending account creation.
- **H3** is fully self-executable: reading permissions and file existence on paths already inside `miktam02`'s own home directory needs no new access. **Actual secret values are never written to `results/` — only existence, permission mode, size, and (for SSH keys) whether a passphrase is set.** This mirrors exp_020's own discipline about not committing the real Tailscale IP: evidence proves the finding without reproducing the sensitive material itself.

---

## H4 — added 2026-07-12 (not in original pre-registration)

**Trigger:** during a risk-modeling pass on this experiment's own findings, the session itself produced two real secret-exposure incidents (careless `grep -n`, and a file-diff-tracking side channel) — evidence that H3's threat model ("what can a compromised process read") has a sibling: what can the *legitimate* automation itself accidentally expose through its own errors. See `RISK_MODEL.md` Part 3 for the full analysis.

**Hypothesis:** two deterministic `PreToolUse` guardrails — `block_secret_reveal.py` (confidentiality: blocks Bash commands that would print known secret-holding files) and `dcg`/destructive_command_guard (integrity/availability: blocks destructive Bash commands) — are both registered and both functionally correct (block the bad case, allow the benign case), and this is re-verifiable rather than a one-time claim.

**Result: H4 CONFIRMED.** `test_h4_agentic_guardrails.sh` — self-executable, no privilege escalation, no secret values recorded — confirms both hooks registered in `~/.claude/settings.json` and both functionally correct on a decoy/benign pair. First run had a harness parsing bug (empty stdout on allow misread as a missing decision, not a real finding) — moved to `results/aborted/` per the append-only convention rather than deleted; corrected run confirms all four checks pass.

**Explicitly not covered by H4** (see `RISK_MODEL.md` for the full residual-risk discussion): the file-diff-tracking exposure side channel, exfiltration via `Write`/`Edit` rather than `Bash`, and anything exp_022 would test.

*Evidence: `test_h4_agentic_guardrails.sh`, `results/h4_agentic_guardrails_2026-07-13T04-25-07Z.json`, `results/aborted/h4_agentic_guardrails_2026-07-13T04-23-58Z_parsing_bug.json`.*

---

## Explicitly noted, not folded into this experiment

A materially different and bigger follow-up: whether a *crafted document fed through the actual pipeline* could induce the LLM or its tools to locate and exfiltrate the same secrets H3 checks for directly — prompt-injection-driven exfiltration, not OS-level permission testing. More relevant to the real threat model for a document-processing system, but a different experiment (candidate: exp_022), not this one.
