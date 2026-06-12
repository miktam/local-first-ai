# Exp 016 — MBP Setup: MLX Inference Server

Hardware target: MacBook Pro 14" M5 Max, 128 GB unified memory.  
This is a new machine — nothing pre-installed.

---

## 1. Prerequisites

```bash
# Verify macOS (MLX requires 13.5+)
sw_vers -productVersion

# Check available disk (~175 GB needed for all four candidate models)
df -h ~

# Confirm Apple Silicon
uname -m   # expect: arm64
```

---

## 2. Install Python + mlx-lm

Use `uv` — faster than pip, no venv juggling.

```bash
# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc   # or open a new shell

# Install mlx-lm into an isolated tool environment
uv tool install mlx-lm

# Verify
mlx_lm.generate --help
```

If you prefer vanilla pip:

```bash
pip3 install mlx-lm
```

---

## 3. Download candidate models

Download in testing order. Each model goes to `~/.cache/huggingface/hub/` automatically.  
Pull one at a time — no need to have all on disk before starting.

```bash
# 1. Control (~13–15 GB) — run this first
mlx_lm.convert --help > /dev/null   # warms HF cache path
python3 -c "from mlx_lm import load; load('mlx-community/gemma-4-26b-a4b-it-4bit')"

# 2. Primary (~50 GB)
python3 -c "from mlx_lm import load; load('mlx-community/Qwen3.5-122B-A10B-4bit')"

# 3. Coding specialist (~40 GB)
python3 -c "from mlx_lm import load; load('mlx-community/Qwen3-Coder-Next-4bit')"

# 4. Optional (~55–65 GB)
python3 -c "from mlx_lm import load; load('mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit')"
```

Or use the HuggingFace CLI (faster with progress bars):

```bash
pip3 install huggingface_hub
huggingface-cli download mlx-community/gemma-4-26b-a4b-it-4bit
huggingface-cli download mlx-community/Qwen3.5-122B-A10B-4bit
# etc.
```

---

## 4. Quick sanity check

Before any benchmark run, confirm a model responds:

```bash
mlx_lm.generate \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --prompt "Reply with one word: hello" \
  --max-tokens 10
```

Expected: short response, no errors, generation starts within a few seconds.

---

## 5. Phase A benchmark: cycling models

For the benchmark, start the server fresh for each model. The server exposes an
OpenAI-compatible API at port 8080.

```bash
# Start server for a given model (replace MODEL_ID as needed)
mlx_lm.server \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --host 0.0.0.0 \
  --port 8080

# In another terminal — quick verification from MBP itself
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ignored","messages":[{"role":"user","content":"Reply: OK"}],"max_tokens":5}' \
  | python3 -m json.tool
```

**From the mini** (verifies LAN reachability):

```bash
curl -s http://mbp.local:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ignored","messages":[{"role":"user","content":"Reply: OK"}],"max_tokens":5}' \
  | python3 -m json.tool
```

Cycle:
1. Start server with model N
2. Run bench script (3 reps — see bench script when produced)
3. `Ctrl-C` server
4. Start server with model N+1

---

## 6. Sleep: keep MBP awake while serving

MLX server dies when the laptop sleeps. While running Phase A:

```bash
# Prevent sleep for the duration of a bench session (Ctrl-C to cancel)
caffeinate -s &
```

For the permanent post-Phase-A setup (smart tier always-on while plugged in), use
System Settings → Battery → "Prevent automatic sleeping when the display is off"
while plugged in. Document the limitation: smart tier is unavailable when the MBP
leaves the home network or runs on battery with display off.

---

## 7. LaunchAgent for the winner (post Phase A)

After Phase A selects the winning model, install a LaunchAgent so the MLX server
starts automatically on login and restarts on crash. Replace `MODEL_ID` with the
winner's mlx-community path.

Create `~/Library/LaunchAgents/com.mlx.serve.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.mlx.serve</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string>
    <string>-c</string>
    <string>mlx_lm.server --model MODEL_ID --host 0.0.0.0 --port 8080</string>
  </array>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <true/>

  <key>StandardOutPath</key>
  <string>/tmp/mlx-serve.log</string>

  <key>StandardErrorPath</key>
  <string>/tmp/mlx-serve.err</string>
</dict>
</plist>
```

Load it:

```bash
launchctl load ~/Library/LaunchAgents/com.mlx.serve.plist
```

Verify it's running:

```bash
launchctl list | grep mlx
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"ping"}],"max_tokens":3}' \
  | python3 -m json.tool
```

Restart / stop:

```bash
launchctl unload ~/Library/LaunchAgents/com.mlx.serve.plist
launchctl load   ~/Library/LaunchAgents/com.mlx.serve.plist
```

Note: LaunchAgent (user-level) vs mini's LaunchDaemon (system-level). The MLX server
only runs while you are logged in. This is correct for a laptop.

---

## 8. Verify full path from mini (Phase B pre-check)

Once the server is running on the MBP, confirm the LAN path works end-to-end:

```bash
# Run from miktam02 (mini)
ping -c 3 mbp.local

curl -s http://mbp.local:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"Reply: OK"}],"max_tokens":5}' \
  | python3 -m json.tool
```

If `mbp.local` doesn't resolve, fall back to the MBP's LAN IP:

```bash
# On MBP
ipconfig getifaddr en0   # WiFi interface
```

---

## Disk layout (reference)

```
~/.cache/huggingface/hub/
├── models--mlx-community--gemma-4-26b-a4b-it-4bit/       ~13–15 GB
├── models--mlx-community--Qwen3.5-122B-A10B-4bit/         ~50 GB
├── models--mlx-community--Qwen3-Coder-Next-4bit/           ~40 GB
└── models--mlx-community--Llama-4-Scout-17B-16E-*/        ~55–65 GB
```
