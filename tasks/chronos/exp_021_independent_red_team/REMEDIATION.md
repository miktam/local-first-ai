# Remediation Plan — exp_020 / exp_021 Findings

*Reference document. Sequencing and impact notes for acting on the findings in `AUDIT_FINDINGS.md` (exp_020) and `H3_FINDINGS.md` (exp_021).*

---

## Publication status & risk acceptance — 2026-09-20

**Decision (risk owner):** rotation of the exposed credentials (`ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `GOG_KEYRING_PASSWORD` and the `gog` OAuth tokens it protects) is **consciously declined as an accepted risk**, not an open action item. This supersedes the "Pending" rows in the Tier 1 tracker below and the "not yet applied" language in `H3_FINDINGS.md` — both predate this decision.

**Rationale:** `miktam-mini` is a single-user, physically-controlled machine with no evidence of compromise; the realistic exploit path (a compromised dependency or crafted document achieving code-execution *as `miktam02`*) reaches these credentials whether or not they are rotated, so rotation does not close the path the finding describes. Storage was hardened where cheap (the Telegram token was reduced from three plaintext locations to one, 2026-07-12) and two agentic guardrails were added this session (`block_secret_reveal.py`, `dcg` — see `RISK_MODEL.md` Part 3). The `gog` account is expendable/non-primary, re-scored to low impact accordingly.

**Residual risk (accepted, stated plainly):** every credential listed here remains technically readable by any process running as `miktam02`. The SSH-key-without-passphrase finding (Tier 3) is likewise open. This is the accepted position, not a mitigated one.

**Portability caveat, stated plainly:** an exfiltrated API key or bot token is usable from any machine, indefinitely, without further access to this box. The acceptance therefore rests on the *low likelihood* of code-execution as `miktam02` here (single user, private mesh network, no churn of untrusted dependencies, no evidence of compromise), not on the impact of a theft being small. A one-off rotation does not close the exposure the finding describes (a new value would land in the same globally-inherited environment); the fix that would is Tier 2 — per-process injection / Keychain / OAuth — which remains open.

**Revisit triggers — this acceptance is void and rotation becomes mandatory if any of the following occurs:** a second user or account gains access to the machine; the machine moves to a different network or its physical custody changes; any dependency or package run as `miktam02` is found to be compromised; or any credential is confirmed to have left the machine (public repo, shipped log, breach).

**Why publishing this is safe:** the exp_021 directory records variable *names* and file *permissions* only — no secret values — verified by a value-shape scan on 2026-09-20 (no `sk-ant-` keys, no bot-token shapes, no key blobs). Publishing documents the *reachability* finding; it does not disclose the credentials themselves.

**Out of exp_021's original scope, flagged not folded in:** the CasaSol bot's `CASASOL_TELEGRAM_TOKEN` (a *different* token from the Nestor `TELEGRAM_BOT_TOKEN` covered here) is supplied from the environment at launch and was never part of exp_021's three-credential scan — its exposure status is unconfirmed and belongs in a separate pass, not retrofitted into this record.

---

## Tier 1 — today, external, only the operator can do this (~15 min)

Rotate all three credentials found exported in plaintext (exp_021 H3). This alone neutralizes everything currently exposed, including in the 12 historical `openclaw.json` backup copies — a revoked credential is worthless even if read later.

- [ ] `ANTHROPIC_API_KEY` — console.anthropic.com, revoke old, generate new
- [ ] `TELEGRAM_BOT_TOKEN` — message @BotFather, `/revoke` or regenerate
- [ ] `GOG_KEYRING_PASSWORD` — rotate; consumer not found in any repo checked (`REPOS/casasol`, `REPOS/pharos`, `REPOS/panacea`, `.openclaw/`) — confirm what depends on it before deciding whether to keep, relocate, or just remove from `.zshrc` if unused

**Sequencing risk:** rotating without immediately updating consumers causes an outage, not just a security fix. Two things break the instant the old value is revoked, before anything downstream is updated:
- Anything reading `ANTHROPIC_API_KEY` (Claude Code, Panacea's `pipeline/llm.py`) stops authenticating until the new key is in place.
- The Telegram bot goes silent until `openclaw.json` has the new token.

**Do Tier 1 live, one credential at a time, immediately followed by updating its one consumer** — not as a batch of rotations followed by a separate cleanup pass.

---

## Tier 2 — this week, fixes the pattern, not just the instance

| Item | Current state | Fix | Real impact |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Exported globally in `.zprofile`, inherited by every process this account runs | Remove the export; use `ant auth login` (OAuth profile, not inherited by every child process) for interactive use | Near-invisible day to day — one-time browser login, then works the same as now |
| Panacea's Anthropic key consumption | `os.environ.get("ANTHROPIC_API_KEY")` in `pipeline/llm.py`, depends on the global export | Wrap Panacea's invocation so the key is injected only for that one process, pulled from Keychain at run time | Not visible in daily use, but is a real code/launcher change, not just a config edit — needs doing before the global export is removed, or Panacea breaks |
| `TELEGRAM_BOT_TOKEN` | Plaintext in the live `openclaw.json`, plus 12 historical backup copies in `~/.openclaw/` | Check whether OpenClaw's config supports an env-var reference or secrets file instead of a literal string; if not, at minimum `chmod 600` the live config and the backups | Depends on what OpenClaw actually supports — not yet investigated |
| `GOG_KEYRING_PASSWORD` | Same exposure pattern as the above | Pending — depends on what it's for (see Tier 1) | Unknown until consumer is identified |

---

## Tier 3 — cheap, independent of the above

**SSH key (`id_ed25519`) has no passphrase.** Checked and confirmed: this key is already used for **unattended automation** — `exp_016`'s two-Mac orchestration scripts (`bench_phase_b.py`, `exp_016b_leaf_executor.py`, `phase_c_orchestrator.py`) SSH/rsync between `miktam-mini` and `miktam-mbp` using it, with no `ssh-agent`/Keychain integration currently configured (no `~/.ssh/config`, `ssh-add -l` reports no agent running).

**Real impact if a passphrase is added without the rest of this:** those automation scripts hang waiting for a passphrase that never arrives in a non-interactive context.

**Correct fix, not just "add a passphrase":**
1. Add a passphrase to the key (or generate a hardware-backed replacement and retire this one).
2. Set up `ssh-agent` + macOS Keychain caching: `ssh-add --apple-use-keychain ~/.ssh/id_ed25519`, and add `UseKeychain yes` / `AddKeysToAgent yes` to `~/.ssh/config`.
3. Type the passphrase once after login; macOS caches it via Keychain from then on, so the `exp_016` automation keeps working unattended exactly as before.

Do steps 1–3 together, in one sitting — not step 1 alone.

---

## Tier 4 — finishing what was already open, not urgent

- [ ] **Screen-lock delay** (currently 3,600s / 1 hour) — only affects physical console access; no effect on Ollama, Panacea, or any background daemon, which run regardless of screen-lock state. Tightening to ~5s means typing the account password sooner after the machine goes idle — the only day-to-day friction in this whole plan that's purely about the human, not automation.
- [ ] **`tail` sudoers scoping** (optional) — real tradeoff, not a free win: the broad grant exists specifically so an agent session doesn't stall on a password prompt it can't answer. Narrowing it risks recreating exactly that stall the next time a legitimate need falls outside the new scope. Only do this with a generously-scoped allowlist, not a reflexively narrow one.
- [ ] **exp_021 H1/H2** — needs a second tailnet device (for H1, curl tests from `miktam-mbp`) and a throwaway non-admin test account (for H2). Not urgent; whenever convenient. See `HYPOTHESIS.md`.

---

## What NOT to do

- Don't rotate all three credentials in a batch and walk away before updating consumers — see the Tier 1 sequencing note.
- Don't add an SSH passphrase without also setting up `ssh-agent` + Keychain in the same sitting — see Tier 3.
- Don't narrow the `tail` sudoers grant reflexively — it exists to solve a specific, real problem (agent sessions can't answer password prompts); undo that problem's solution only with something that still solves it.

---

## Status tracker

| Tier | Item | Status |
|---|---|---|
| 1 | Rotate `ANTHROPIC_API_KEY` | **Accepted risk — rotation declined by owner 2026-09-20** (see Publication status above) |
| 1 | Rotate `TELEGRAM_BOT_TOKEN` | **Accepted risk — rotation declined by owner 2026-09-20**; storage reduced to one location 2026-07-12 |
| 1 | Rotate `GOG_KEYRING_PASSWORD` | **Accepted risk — rotation declined by owner 2026-09-20**; consumer identified (the `gog` file-backed keyring holding a real Google account's OAuth tokens), impact re-scored low (expendable account) |
| 2 | Migrate Anthropic auth to `ant auth login` | Pending |
| 2 | Panacea Keychain-wrapper launcher | Pending |
| 2 | Telegram token out of plaintext config | Pending — needs OpenClaw config investigation |
| 3 | SSH passphrase + agent/Keychain setup | Pending |
| 4 | Screen-lock delay | Pending — command given twice, outcome unconfirmed |
| 4 | `tail` sudoers scoping | Pending — optional |
| 4 | exp_021 H1/H2 | Pending — needs cooperative access |
