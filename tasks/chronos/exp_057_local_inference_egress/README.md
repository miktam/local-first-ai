# exp_057 — does local inference leave the machine?

Watches a model runtime's **process tree at the OS level** while it loads, idles and generates, and the **whole
machine** beside it, to test the local-first claim that nothing leaves the machine during local inference. Prompted
by a report that a third-party **Qwen** appeared to open outbound connections during local use.

The point that makes it falsifiable: **weights are inert data.** A connection comes from the *runtime* or a *harness*,
never the tensors. So the instrument compares a **load window**, an **idle window** (model loaded, not generating) and
a **driven window**, and anything external that appears only while generating is the finding.

- `HYPOTHESIS.md` — registration, subjects, method, controls, H1–H4, limits, and the pre-registration review.
  Registered 2026-10-28, before any run.
- `data/prompts.json` — the frozen, benign prompt set (sha256 in HYPOTHESIS.md).
- `kit/egress_watch.py` — two samplers without root: `Watcher` (`lsof` over the Ollama daemon and its runner children
  every 0.2 s; the runtime is not Python, so the exp_054 audit hook cannot see it) and `MachineWatcher` (`nettop`
  over every process every second, with byte counts). Neither resolves a name.
- `kit/drive.py` — sends the prompts to Ollama (`/api/generate`, no tools; `/api/embed` for the embedding model),
  records each generation's window and output.
- `kit/control_pos.py` — positive control: one connection to `example.com:80` that the watcher must catch; run once
  directly (C_pos) and once from a grandchild spawned mid-window (C_tree).
- `kit/hook/sitecustomize.py` — the exp_054 Python audit hook, for the driver's loopback ground truth (C_loop).
- `kit/run57.py` — orchestrates the controls, then per model: load → idle (P0) → driven (P1), both samplers on.
- `kit/score57.py` — classifies every endpoint, evaluates the controls and H1–H4, lists what the rest of the machine
  did, and reads an optional packet capture as text (`--pcap-text`, experimental parser).

## Running it

```bash
cd tasks/chronos/exp_057_local_inference_egress
ollama list                      # all ten subjects must be present; the run pulls nothing
python3.14 kit/run57.py runs/$(date -u +%Y%m%dT%H%M%SZ)
python3.14 kit/score57.py runs/<that dir>
```

Optional full-capture arm, in a second pane, from an account with a sudo grant on a fixed-argument wrapper around
`tcpdump -i pktap,all -P -n -s 128 -w <file>` (headers only, process metadata kept); then
`tcpdump -n -k NP -r <file> > capture.txt` and `score57.py runs/<dir> --pcap-text capture.txt`.

**Status: registered, not yet run.** The run is Andrei's to start (it drives the shared production Ollama and may
evict the bot's resident models). See the HYPOTHESIS for the limits — most importantly that a clean result means
"nothing left the machine under this workload," never "this model is safe."
