# miktam-mini Hardening Audit — Findings Record

*Companion to `HYPOTHESIS.md` and `results/`. Read-only audit conducted 2026-07-08, cross-checked empirically wherever a static config read could be misleading (see the `cp`/`pmset` correction below — this is the whole reason the audit re-tests rather than trusts `sudo -l -n` output at face value).*

---

## Already in good shape — no action

| Control | Status |
|---|---|
| FileVault-independent controls: firewall | Enabled |
| SIP | Enabled |
| Gatekeeper | Enabled (assessments enabled) |
| Automatic update checking | On |
| Automatic security/OS update *install* | On (`AutomaticallyInstallMacOSUpdates=1`, `CriticalUpdateInstall=1`, `AutomaticDownload=1`) |
| SSH key-based auth configured | Yes — 3 keys in `~/.ssh/authorized_keys` |
| Tailscale device roster | Reviewed — all devices (`macbook-pro-tomasz`, `polazlmac0004`, `polazlmac0006`) confirmed known and authorized |
| `pharos-deploy` NOPASSWD script | Reviewed by source — fixed hardcoded paths, no attacker-influenced input, bounded to one repo + one plist. Low risk as written. |

---

## Fixed as part of this task

**Ollama was open to the entire Tailscale network with zero authentication** (`OLLAMA_HOST=0.0.0.0:11434`, confirmed exploitable in exp_020 H2 — an unauthenticated POST returned a real model response). Fixed by rebinding Ollama to `127.0.0.1:11434` and putting a token-authenticated Caddy reverse proxy on the Tailscale-facing side (`<tailnet-ip>:11435`). Remote/field-device access now requires `Authorization: Bearer <token>`; token lives at `~/.config/ollama-proxy/token`, mode 600.

**Applied and verified 2026-07-08.** Post-fix checks: the old port (`<tailnet-ip>:11434`) now times out from the Tailscale-facing address entirely (connection refused/timeout — confirmed via `curl` from another point on the tailnet); an unauthenticated request to the new proxy address returns HTTP 401; the identical request with the correct bearer token returns HTTP 200. Ollama remains reachable at `127.0.0.1:11434` from the machine itself — expected and required, since the proxy forwards to it locally. Full evidence in `results/h2_ollama_exposure_post_*.json`.

---

## Documented as accepted risk — not changed

Per explicit decision: sudoers stays as-is (agentic-coding convenience), FileVault stays off (always-on-server uptime).

**sudoers — corrected picture (exp_020 H1/H3, see `results/`):**

| Grant | Actually passwordless? | Severity | Notes |
|---|---|---|---|
| `tail` (unscoped) | **Yes — confirmed** | **High** | Reads any file `miktam02` couldn't otherwise read, including `/etc/sudoers` itself (demonstrated: direct `cat /etc/sudoers` denied, `sudo -n tail /etc/sudoers` succeeded). Root-level read, no write. |
| `shutdown` | Yes — confirmed | Low (availability only) | Can take the always-on node offline without a password; no data exposure. |
| `lsof` | Yes — confirmed | Low | Standard debugging/visibility tool. |
| `asitop` / `powermetrics` | Yes — confirmed | Low | Telemetry only. |
| `cp` | **No — refuted** | — | Appeared unrestricted in the static `sudo -l -n` listing; direct invocation testing shows it's actually password-gated. Likely a line-wrap artifact in how the listing rendered two adjacent rules as one. **This is the single most important correction in this audit** — the thing that looked like the worst finding (arbitrary root file write) isn't real. |
| `pmset` | No — refuted | — | Same pattern as `cp`: appears broad in the listing, password-gated in practice. |
| `tailscaled`/`ollama` launchctl bootstrap/bootout | Not independently re-tested (already relied upon daily per documented restart procedure in CLAUDE.md) | Low | Scoped to exact plist paths, not general `launchctl` access. |

**The corrected finding for the record: the real risk is unrestricted `tail`, not `cp`.** It grants confidentiality-breaking read access (can see the full sudoers config and anything else root-only on the box) but not a write/escalation primitive. Given the accepted-risk decision, no sudoers change is made — but anyone reviewing this machine's security posture should know the actual exploitable surface is narrower than the raw `sudo -l` dump implies in one direction (no `cp`/`pmset` escalation) and real in another (full read via `tail`).

**FileVault is off.** Client documents processed by this pipeline sit unencrypted at rest. Accepted for now given always-on-server uptime requirements (FileVault requires unlock after reboot/power loss). **Revisit trigger: before onboarding a second client's corpus, or if the machine's physical location/custody changes.**

---

## New findings not in the original hardening scope

**SSH (`Remote Login`) is listening on all interfaces, not just Tailscale** (`launchd ... TCP *:ssh (LISTEN)` on both IPv4 and IPv6) — confirmed currently in legitimate active use by an authorized tailnet device (`polazlmac0004`). Key-based auth is configured (3 keys in `authorized_keys`). **Resolved 2026-07-08**: confirmed `PasswordAuthentication no` in `/etc/ssh/sshd_config` — key-only login is enforced, so the broad interface exposure is not a brute-force surface. No further action needed.

**Account boundary confirmed 2026-07-08, and it's a real compensating control, not an accident.** `miktam` is the administrator account (interactive human login). `miktam02` is a **standard, non-admin account** — it holds no general sudo rights of its own, only the specific narrow NOPASSWD grants added for agentic-coding convenience — and it's the account Ollama and this automation actually run under. That means the confirmed `tail` escalation (above) is scoped to what a standard account plus its explicit grant list can reach, not to full administrator access. Least-privilege separation between the human's admin session and the automation's runtime account, already in place before this audit — worth stating plainly as a thing that's *right*, not just a gap list.

**Screen lock delay is 3,600 seconds (1 hour).** The screen doesn't require a password until a full hour after sleep/screensaver activation — lax for a machine handling confidential documents, though it only matters for someone with physical console access (not a remote-network risk). Setting this to near-immediate is safe and low-friction (doesn't affect background services or the pipeline's continuous operation — daemons keep running regardless of screen lock state); recommend tightening this to a short delay (e.g. 5 seconds) as a follow-up, since it wasn't part of the original negotiated scope but has essentially no downside.

---

## Summary for the non-technical read

Three real things were found and fixed or clearly written down: the AI model's door was unlocked to anyone on the private network (now locked with a key); one system permission quietly let a background process read *any* file on the machine without asking (left as-is by choice, but now precisely known rather than guessed at); and the actual hard-disk encryption is still off (a known, deliberate gap, not an oversight). One early finding that looked like the worst problem — a permission that seemed to allow overwriting any file as an administrator — turned out, on actually testing it rather than just reading the configuration, not to be real. That correction is itself the point: check by doing, not by reading a list.
