# exp_057 result: inference opened nothing; the runtime phones home on a timer

**Run:** `runs/20261028T050604Z`, 2026-10-28 05:06–05:35 UTC, 29 minutes, on the Mac mini. Ollama 0.40.2 on macOS 26.5.1, started by Andrei with the registered defaults (every subject, 30 s idle, every prompt). All three controls passed, so the run is valid:

- **C_pos:** the tree sampler caught the control process's connection to `example.com:80`.
- **C_tree:** the same connection, made by a grandchild spawned half a second into the watch, was caught from the grandchild's PID.
- **C_loop:** the driver's call to `127.0.0.1:11434` was seen by `lsof` and by the hook, and the hook shows no other network act by the driver.

The full-capture arm did not run (no sudo grant existed yet), so DNS queries are unseen and the samplers stand with their stated limits.

## Verdicts

| | Claim | Result | Verdict |
|---|---|---|---|
| H1 | Plain inference stays on the machine | Nine of ten subjects opened no non-loopback endpoint at load, idle or while generating. During `qwen3.5:35b`'s driven window the Ollama **daemon** (PID 562, not the runner) opened one TCP connection to `34.36.133.15:443`, the address `ollama.com` resolves to | **fail** |
| H2 | No endpoint tracks a generation | That connection was first seen at 05:26:39.8, inside prompt 2's window ("What is the capital of Spain…"), whose answer was a plain fact | **fail** |
| H3 | The runtime's own background egress is named | At load and at idle: nothing, for every subject, including the two whose on-disk format was upgraded at load. The one connection is **named**: `ollama.com`, 2,001 bytes out and 7,639 bytes in, made by the daemon's **model-recommendations refresh** (below) | named |
| H4 | Tool-call output is inert here | No answer of any subject matched the tool-call or search-request markers, including the five prompts written to tempt one. Nothing executed anything | recorded |

**The verdicts stand as registered.** H1 and H2 were written to detect an appearance, not a cause, and they detected one. The explanation below sits beside the fails, not in place of them. The registered design could not have told a scheduled refresh from a reaction to a prompt: a 30 s idle baseline cannot see a four-hour timer, and with generation covering most of the driven window, any timer that fires lands inside some prompt's window. One occurrence cannot separate "tracks a generation" from "coincided with one".

## What the connection was

The daemon's own log settles it, to the millisecond:

| Time (UTC) | Source | Event |
|---|---|---|
| 01:00:52.481 | `ollama.log` | `model_recommendations.go:177 "model recommendations cache sleep scheduled" wait=4h25m47s` (the previous refresh), so the next was due at 05:26:39.5 |
| 05:26:39.773 | `ollama.log` | the same line, `wait=3h58m35s`, `consecutive_failures=0`: the refresh ran and succeeded |
| 05:26:39.831 | tree sampler | first sight: `ollama 562 TCP 192.168.18.241:51636->34.36.133.15:443 (ESTABLISHED)` |
| 05:26:39 | `~/.ollama/cache/model-recommendations.json` | rewritten, 2,041 bytes, a list of six recommendations |
| 05:26:40.394 | machine sampler | the same flow, interface `en1`, 7,639 bytes in, 2,001 out; no further bytes; still open when the window ended at 05:31:47 and gone by the next |

The daemon's three logged failures name the request: `GET https://ollama.com/api/experimental/model-recommendations?ts=<unix seconds>`, a timestamp and nothing else visible. The refresh also runs at every daemon start, retrying after about 4.5 minutes when the network is not up yet. The log holds 372 such refreshes since 2026-03-05, every 3.5 to 4.6 hours, three of them failed: about six contacts with `ollama.com` a day, whatever the machine is doing. The daemon started with `Ollama cloud disabled: false`, the default. The timer fell in `qwen3.5:35b`'s window by arithmetic, not by anything that model or prompt did.

What left is 2,001 bytes: consistent with a TLS handshake plus one HTTP request. Whether the request carries an identifier of this instance (its public key, its model list) cannot be told from the bytes, and the documentation does not say. What came back is the recommendations list.

## The background model upgrade (descriptive)

Two subjects had manifests from before 0.40.2 (`qwen3.6:35b`, pulled four months ago, and `qwen3.5:35b`, seven). On their first load the daemon ran a "local compat GGUF migration" (30.3 s and 25.2 s), which the 0.40.2 notes call upgrading models "in the background the first time you run them", keeping the original as a backup. Both migrations ran inside watched windows: **no network connection from the tree** during either, to the samplers' resolution. The cost was on disk: 23 blobs, 47.8 GB written; the backups are kept until a future release removes them.

## Per subject

| Subject | Load | Idle | Driven (10 prompts) | External endpoints |
|---|---|---|---|---|
| `qwen3-4b-b11577:q4km` | 4.9 s | 30 s | 72 s | none |
| `qwen3-4b-b11577:q8` | 4.3 s | 30 s | 116 s | none |
| `qwen3-4b-b11577:q4km-imat` | 2.2 s | 30 s | 111 s | none |
| `hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M` | 2.2 s | 30 s | 94 s | none |
| `hf.co/unsloth/Qwen3-1.7B-GGUF:Q4_K_M` | 1.2 s | 30 s | 37 s | none |
| `qwen3.8:27b` | 15.0 s | 30 s | 314 s | none |
| `qwen3.6:35b` | 18.4 s (migrated) | 30 s | 109 s | none |
| `qwen3.5:35b` | 14.3 s (migrated) | 30 s | 385 s | **ollama.com, by the daemon, at 05:26:39** |
| `gemma4:26b` | 10.4 s | 30 s | 117 s | none |
| `nomic-embed-text` (`/api/embed`) | 0.6 s | 30 s | 0.4 s | none |

No prompt errored. Load times are the warm-up call; the two migrations continued into the idle windows, which were watched.

## The rest of the machine (descriptive)

While the models generated, the machine sampler saw twelve other processes with non-loopback flows, none in the Ollama tree: the two Telegram bots (Python), Claude Code sessions, the VS Code server (`node`, seven endpoints, one of them the cloud-metadata address `169.254.169.254:80`), `ssh` and `sshd-session`, `tailscaled`, and Apple's `apsd`, `geod`, `mobileassetd`, `syspolicyd` and `trustd`. Their addresses are in the machine-sampler logs, which are **held back** from this repository: they map every service the machine talks to, including addresses of other accounts' sessions, and nothing in this experiment's claims needs them public. `score.json` is the publication form (Amendment 1): the tree's own flows verbatim, other processes' addresses replaced by counts. The held-back files, sha256 (first 12):

`C_pos.machine.jsonl` eafaa7bfdd42
`C_tree.machine.jsonl` 4ce46066be83
`drive_gemma4:26b.machine.jsonl` ff97aab593b7
`drive_hf.co_unsloth_Qwen3-1.7B-GGUF:Q4_K_M.machine.jsonl` 0a72dbb70574
`drive_hf.co_unsloth_Qwen3-4B-GGUF:Q4_K_M.machine.jsonl` 0cc30042f5bd
`drive_nomic-embed-text.machine.jsonl` 21730fd3f166
`drive_qwen3-4b-b11577:q4km-imat.machine.jsonl` 9e46614dc682
`drive_qwen3-4b-b11577:q4km.machine.jsonl` 8dc96f7b314e
`drive_qwen3-4b-b11577:q8.machine.jsonl` f428898755ec
`drive_qwen3.5:35b.machine.jsonl` 2118834afa8c
`drive_qwen3.6:35b.machine.jsonl` 709fce03e617
`drive_qwen3.8:27b.machine.jsonl` 9e1a7658bc39
`idle_gemma4:26b.machine.jsonl` cef9bc7d4652
`idle_hf.co_unsloth_Qwen3-1.7B-GGUF:Q4_K_M.machine.jsonl` 4620a0530263
`idle_hf.co_unsloth_Qwen3-4B-GGUF:Q4_K_M.machine.jsonl` 0eea6e0d7565
`idle_nomic-embed-text.machine.jsonl` 69b7f2412e5b
`idle_qwen3-4b-b11577:q4km-imat.machine.jsonl` b00f19313fae
`idle_qwen3-4b-b11577:q4km.machine.jsonl` da8e68702242
`idle_qwen3-4b-b11577:q8.machine.jsonl` 88ba8388d310
`idle_qwen3.5:35b.machine.jsonl` 0beebd298cfa
`idle_qwen3.6:35b.machine.jsonl` 07883fe4a79c
`idle_qwen3.8:27b.machine.jsonl` b780d129d88b
`load_gemma4:26b.machine.jsonl` ba236a92144e
`load_hf.co_unsloth_Qwen3-1.7B-GGUF:Q4_K_M.machine.jsonl` b895702667d8
`load_hf.co_unsloth_Qwen3-4B-GGUF:Q4_K_M.machine.jsonl` 81c6a2592aa2
`load_nomic-embed-text.machine.jsonl` e35efd674abb
`load_qwen3-4b-b11577:q4km-imat.machine.jsonl` 7a305c2c5d8a
`load_qwen3-4b-b11577:q4km.machine.jsonl` 1f9bff6da5f2
`load_qwen3-4b-b11577:q8.machine.jsonl` 94d4891cbadf
`load_qwen3.5:35b.machine.jsonl` db826fc1f532
`load_qwen3.6:35b.machine.jsonl` 6d1eccf00b4e
`load_qwen3.8:27b.machine.jsonl` 0d22b92e038a

This is the picture a review-machine dashboard has to show rather than assume quiet.

## Limits, as registered and as met

- **Resolution.** `lsof` at 0.2 s and `nettop` at 1 s. A connection shorter than that could have been missed; the capture arm that would close this did not run.
- **DNS unseen.** The daemon must have resolved `ollama.com` through the machine's resolver (the LAN router and Tailscale's MagicDNS). Only the capture arm would show it.
- **Payload unseen.** The endpoint and the byte counts are known; the request's content is not.
- **One occurrence.** The cadence comes from the daemon's log, not from this run.
- **Honest null, restated.** Nine subjects showed nothing under this workload. That is not "these models are safe".

## SO WHAT

- **"Nothing leaves the machine during local inference" is false for Ollama 0.40.2 as installed.** With cloud features on, the default, the daemon contacts `ollama.com` about six times a day, generating or not. It is the runtime, not the weights and not the prompt: layer 1 of the hypothesis. Every place we wrote that local inference sends nothing needs the qualifier: *the model sends nothing; the default runtime does.*
- **The report that prompted this is not what we saw.** No Qwen subject, nor any subject, opened a connection. The harness layer, an app executing a model's search request, is still untested and is the follow-up experiment.
- **The fix is a flag, and it is unmeasured: Amendment 2 is registered to measure it** (HYPOTHESIS.md; runner `kit/idle57.py`). `OLLAMA_NO_CLOUD=1` (or `"disable_ollama_cloud": true` in `~/.ollama/server.json`) plus a restart; the log must then read `Ollama cloud disabled: true`. The documentation does not say whether the recommendations refresh is covered by that flag. A follow-up, registered before it runs: set the flag, restart, watch the tree for longer than the longest observed interval (five hours) with no generation, and check the log for refresh lines. The production environment is Andrei's to change.
- **For the egress dashboard and the Exposure Review annex.** The runtime baseline must be longer than the longest timer; a 30 s idle window proves nothing about a four-hour one. Until the flag is set and measured, the annex lists the daemon's `ollama.com` refresh as known egress of the review machine.
- **The background upgrade** made no network call but wrote 48 GB of backups. Worth knowing before the next large pull.
