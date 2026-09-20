# H3 Findings — Secret Exposure via Ordinary File Permissions

*No sudo, no privilege escalation, no exploit — this is what's reachable by anything already running as the account that runs Ollama and this automation. See `results/h3_secret_exposure_*.json` for the machine-readable evidence (names/metadata only, no values).*

---

## H3: confirmed

Four things are directly reachable with zero elevation:

1. **The SSH private key (`id_ed25519`) has no passphrase.** Any code running as this account can use it immediately for outbound authentication — no additional secret needed, no barrier beyond file read.
2. **`ANTHROPIC_API_KEY` is exported in a shell startup file and present in the live environment of every session.** Readable via a plain `cat` of the startup file, or trivially via `env` from inside any process — no need to even know the variable name in advance.
3. **`TELEGRAM_BOT_TOKEN` is exported the same way** — full control of the associated bot for anything that reads it.
4. **`GOG_KEYRING_PASSWORD` is exported the same way** — scope of what this unlocks wasn't investigated further here, but it's the same exposure pattern.

Git credential storage came back clean — no plaintext credentials file, no `.netrc`, no credential helper configured. Auth for git is presumably the same SSH key covered above.

---

## Why this is the finding that matters most so far

Exp_020 tested what a privileged escalation (`sudo`) could reach. This tests what's reachable with **no escalation at all** — just running as the account that already, legitimately, runs Ollama and the document pipeline. That's a much lower bar for an attacker to clear, and it's the realistic path: a compromised dependency, a malicious package pulled in by tooling, or (the sharper version, not tested here — see exp_022 in the parent `HYPOTHESIS.md`) a crafted document that induces the pipeline's own LLM-driven tooling to read and act on these files. None of that needs a sudoers misconfiguration. It just needs code execution as the account that was always going to have it.

The exp_020 `tail` finding was a real gap, but it was read-only against system files. This is read-only against **live, billable, bot-controlling credentials**, using nothing more than `cat`.

---

## Recommended remediation (as originally written — see disposition below)

> **Disposition, 2026-09-20:** rotation was subsequently **declined as an accepted risk by the risk owner** — see `REMEDIATION.md` § "Publication status & risk acceptance" for the rationale and residual risk. The recommendations below stand as what the experiment recommended at the time; they are not all what was done.

1. **Rotate `ANTHROPIC_API_KEY` and `TELEGRAM_BOT_TOKEN`** as a precaution. This session's tool calls have had them in-environment the whole time; they were never intentionally read or echoed by anything in this experiment (the test script only ever wrote variable *names* to disk, confirmed in `results/`), but "never intentionally read" is not the same guarantee as "could not have been read."
2. **Move both out of shell startup files** into something that isn't globally exported to every process by default — macOS Keychain (`security add-generic-password` + a small wrapper to fetch on demand), or a secrets tool, rather than `export X=... ` in `.zshrc`/`.zprofile`.
3. **Add a passphrase to the SSH key**, or migrate outbound auth to a hardware-backed key (Secure Enclave / YubiKey) so the private key file alone isn't sufficient.

---

## Status

H3: **confirmed**. H1 and H2 remain blocked on cooperative access (separate device, throwaway test account) — see `HYPOTHESIS.md`.
