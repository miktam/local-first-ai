# exp_057: does local inference leave the machine?

**Registered 2026-10-28, before any measured run.** Kit: `kit/`. Prompt set frozen by the hash at the end. The run is Andrei's to start (it drives the shared production Ollama; the full-capture arm needs a sudo grant). Drafted in the morning, reviewed and strengthened in the afternoon before this registration; the review is recorded at the end.

## Why

A colleague reported that a local **Qwen**, run on his own machine, appeared to open outbound connections during use — endpoints in China, then an OVH host in France — and that "it just searches for data in a search engine." That is the exact failure the local-first thesis rules out: if inference is local, nothing should leave the machine. We assert this in Aprimerose material and on localfirstai.eu, and exp_054 verified it for the *Exposure Review's own code*. We have never watched a **third-party model's runtime** while it generates. This experiment does.

## The distinction that makes it falsifiable

Model weights are inert data. A `.gguf` file holds tensors; it cannot open a socket. An outbound connection during "local inference" can only come from one of three layers, and the experiment is built to tell them apart:

1. **The runtime** — the process that loads the weights and runs the math (here Ollama's daemon and the runner it spawns per model). A runtime can check for updates, fetch models, or send telemetry. These are properties of the *program*, not the model.
2. **A harness around the runtime** — a chat app or agent loop that reads the model's output, sees a tool-call or a search request in it, and *executes* that call. "It searches a search engine" is this layer. The model emits text; something else makes the request.
3. **The weights** — which do nothing on their own.

So the instrument watches the **runtime's process tree at the OS level**, not just Python, and compares an **idle window** (runtime loaded, no generation) against a **driven window** (generation in flight). Anything external that appears only while generating is the finding. The moment the model is **loaded** is watched too, because an update check or a manifest fetch would most plausibly happen then and nowhere else.

## Subjects

Run through the production Ollama (0.40.2), each as the bot calls a model (`/api/generate`, stream, no tools, temperature 0.3, context 8192):

- **The flagged family:** `qwen3-4b-b11577:q4km`, `:q8`, `:q4km-imat`, `hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M`, `hf.co/unsloth/Qwen3-1.7B-GGUF:Q4_K_M`, and the large `qwen3.5:35b`, `qwen3.6:35b`, `qwen3.8:27b`.
- **In-house control:** `gemma4:26b` (the live bot model) and `nomic-embed-text` (the bot's embedding path, driven through `/api/embed`; its "answer" is the vector's length).

All ten were present in `ollama list` on 2026-10-28, so the run pulls nothing. The large Qwen models are loaded one at a time to fit the mini; a model that will not fit is recorded as skipped, not forced. Loading them may evict the bot's resident models (`gemma4:26b`, `gemma4:e4b`); that is accepted, since the bot has been out of use since 2026-10-10 and reloads on its next request.

## Method

- **Process tree.** Before each sample, resolve the Ollama daemon PID and every descendant (`pgrep -P`, repeated), plus the runner by its command line (`libexec/lib/ollama/` on 0.40.2, `ollama runner` on older builds). The watch set is that whole tree, re-resolved each sample, so a runner spawned mid-window is caught.
- **Tree sampler (no root, always runs).** `kit/egress_watch.Watcher` runs `lsof -nP -i -a -p <pids>` every 0.2 s over the tree and records every internet socket with its remote endpoint and the owning command. This is the exp_054 cross-check promoted to the primary instrument, because the runtime is not Python.
- **Machine sampler (no root, always runs).** `kit/egress_watch.MachineWatcher` runs `nettop -n -x -L 1` every second over every process of every user, with interface, state and byte counts per flow. It covers the tree sampler's blind spot: a connection made by a process outside the tree (through launchd or XPC, or by a runner that is not parent-linked) is still seen and named. Flows are split into the tree's and other processes'. Descriptive: 1 s floor, attribution by process name and PID.
- **Python audit hook (the driver).** The driver loads exp_054's `sitecustomize.py` hook, so its own call to Ollama is logged. This gives the loopback ground truth: the request to `127.0.0.1:11434` must appear, and nothing else may.
- **No name resolution by the instruments.** `lsof -n`, `nettop -n`: watching makes no DNS query of its own, and the endpoints are recorded as addresses.
- **Full-capture arm (root, optional).** `tcpdump -i pktap,all -P -n -s 128 -w <file>` records every packet on every interface with the sending process's name and PID (Apple's pktap metadata), headers only. It is packet-level ground truth that a 0.2 s sampler cannot miss a short connection, and it sees DNS queries. It runs from an account holding a sudo grant on a fixed-argument wrapper, never on bare `tcpdump` (whose `-z` and `-w` make a NOPASSWD grant a root shell). Read back with `tcpdump -n -k NP -r <file>` and handed to the scorer as `--pcap-text`; that parser is marked experimental until it has met a real capture. Without the arm, the samplers stand with their stated limits.
- **Windows, per subject:**
  0. **load** — the one-token warm-up that loads the model and spawns its runner, watched.
  1. **P0 idle** — 30 s watched with no generation. Establishes the runtime's own background egress.
  2. **P1 driven** — the 10 frozen prompts sent in sequence, watched throughout, each generation's wall-clock window recorded so an endpoint can be tied to a prompt.
- **Session controls** run once, before the per-model loop.

## Controls (if any fails, the run is void)

- **C_pos** (the watcher sees an external connection). `kit/control_pos.py` opens one TCP socket to `example.com:80` and closes it, watched at the measurement interval (0.2 s). The tree sampler must record a non-loopback endpoint. Proves a real phone-home would not slip past.
- **C_tree** (the watcher sees a descendant spawned later). The same connection, made by a Python process that a watched `sh` spawns half a second after the watch starts. The endpoint must be recorded from a PID other than the root's. Proves the descendant expansion that must catch a runner spawned mid-window.
- **C_loop** (the watcher sees the local call, and the driver does nothing else). During P1 the driver's request to `127.0.0.1:11434` must appear in the Python hook log *and* as a loopback socket in the tree sampler, and the hook log must hold no non-loopback connection or resolution by the driver. Proves the instrument is pointed at the right process, and that the driver cannot be the source of any external endpoint seen.

## Hypotheses

| | Claim | Passes if |
|---|---|---|
| **H1** | Plain inference stays on the machine | for every subject, no non-loopback endpoint (address and port) is opened by the runtime tree during P1 that was not already present at P0 idle |
| **H2** | No endpoint tracks a generation | no external connection's first appearance falls inside any single prompt's generation window |
| **H3** | The runtime's own background egress is named | every non-loopback endpoint seen at load or at P0 idle is enumerated and attributed to a process and a known purpose (update check, registry/CDN, telemetry) — descriptive, this is the "runtime, not model" separation |
| **H4** | Tool-call output is inert here | for each subject, whether its answers to the search-tempting prompts contain tool-call / search-request structure is recorded (coarse text markers); nothing in this kit executes any such call — descriptive |

**Descriptive also.**
- Endpoints the machine sampler attributes to processes outside the tree, per window: what else on the machine was talking while a model generated.
- Any DNS resolution the runtime makes and the names it asks for (visible only with the full-capture arm).
- Exit codes and per-prompt latency.
- Whether a model that *emits* a search request differs from one that does not.
- Endpoints the full-capture arm shows that neither sampler saw.

## Limits

- **Sampling gap.** `lsof` at 0.2 s can miss a connection opened and closed inside 200 ms; `nettop` at 1 s more so. The `tcpdump` arm closes this; without it, H1/H2 hold only to the sampler's resolution, and the result says so.
- **Payload opacity.** For any connection seen, the instrument records the endpoint, not the bytes. It proves *that* a call happened and to where, not what was in it. The capture arm keeps 128 bytes per packet: headers, not content.
- **Attribution by name.** The machine sampler names a process; it does not prove that process acted for the runtime.
- **Runtime scope.** Only the Ollama runtime is tested. LM Studio, a bare `llama.cpp` server, MLX and any desktop chat app are out of scope here; the method ports to each but the finding does not. Two unrelated `llama-server` processes from a separate llama.cpp build run on this machine; they are outside the tree and appear, if at all, among the "other processes".
- **Harness scope.** No tool executor is wired. The experiment can see a model *emit* a search-shaped request; it cannot, and does not, test what a given third-party app would *do* with it. That is the layer most likely behind the report and is named as the next experiment, not answered here.
- **Honest null.** A model that phones home only on certain prompts, only after N turns, or only on a network it recognises may show nothing in this window. A clean result is "nothing left the machine under this workload," never "this model is safe."

## Frozen input (sha256, first 12)

prompts.json 459084e62bbe

## Reviewed before registration (2026-10-28, afternoon)

The morning draft was read against the running machine before it was committed. What changed, and why:

1. **The load window is watched.** The draft watched idle and driven windows only; the warm-up that loads the model and spawns its runner ran unwatched, and that is the most plausible moment for a runtime's update check or manifest fetch. Now `load_<model>` is a step of its own.
2. **A second, machine-wide instrument.** `nettop` over every process, split into tree and other. The draft could not see a connection made outside the tree. Descriptive.
3. **A control that tests the mechanism relied on.** C_pos watched a PID it was handed, at 0.1 s; the measurement relies on *descendant expansion* at 0.2 s. C_tree spawns the connecting process as a grandchild mid-window, and C_pos now runs at the measurement interval.
4. **C_loop reads the hook log it promised to.** The draft's scorer checked `lsof` only. Now both must agree, and the driver's log must show no act but the loopback call.
5. **Runner resolution on 0.40.2.** The draft's by-name patterns (`ollama runner`, `ollama_llama_server`) match nothing on this build; the runner is `libexec/lib/ollama/llama-server`, a child of the daemon. It was caught through descendants only. The path pattern is added.
6. **`nomic-embed-text` can run.** It was a registered subject but the driver spoke only `/api/generate`, which an embedding model refuses; it would have been "skipped". The driver now uses `/api/embed` for it.
7. **Endpoints are keyed by address and port**, not address alone. Byte counts are recorded where `nettop` gives them.
8. **The capture arm is specified**: pktap with process metadata, headers only, through a fixed-argument wrapper, read back as text.
9. Subjects confirmed present; eviction of the bot's models accepted; the follow-up experiment is numbered at its own registration, not here.

**Dev before registration.** Morning: the watcher and positive control were smoke-tested once (a connection to `example.com:80` caught; the daemon PID resolved; no model driven). Afternoon, after the changes above: the whole kit ran once with smoke-test flags (`--idle 3 --prompts 2`) on `gemma4:e4b`, which is **not a subject** and was already loaded. 14 s; C_pos, C_tree and C_loop passed; the tree opened no external endpoint in any window; the machine sampler listed seven other processes with external flows (the bots, an editor server, ssh sessions, the tailnet daemon, Apple push), none of them in the tree. The parsers were unit-checked against live `nettop` output and synthetic `tcpdump -k` lines. The registered run uses the defaults: every subject, 30 s idle, every prompt.

## Next experiment (named, not run here)

A follow-up, numbered when it is registered, would wire a logged, sandboxed harness and measure layer 2: given a model's tool-call output, does a representative local app execute it, and to where. That is where "it searches a search engine" is actually tested.

## Amendment 1 (2026-10-28, after the run, before publication)

The machine sampler's logs (`logs/*.machine.jsonl`) hold every process's remote addresses, including other accounts' sessions and the machine's tailnet address. They are held back from the public record and hashed in RESULT.md. `score57.py --public` writes the publication form of `score.json`: the tree's own flows verbatim, other processes' addresses replaced by counts. No measurement, control or verdict changes; re-scoring the published folder reproduces everything but the other-process section, which needs the held-back files.
