# exp_059 — the phone arm (and control C2), built on the MacBook

Two Swift files added to llama.cpp's own iOS example at the pinned commit. `KevBatch.swift` is the readout
(the server's `/v1/systemone` for a `kev` model, with the public API); `Exp059View.swift` is the screen and the batch loop.
Nothing here needs the network; the app has no network code.

## 1. The engine at b11711, as an Xcode framework

```bash
mkdir -p ~/models && cd ~/models
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp-b11711
cd llama.cpp-b11711 && git checkout b11711 && git rev-parse --short HEAD        # expect 9d1e036
./build-xcframework.sh                                                          # 10–20 min; needs Xcode 16 or newer
ls build-apple/llama.xcframework                                                # ios-arm64, ios-arm64-simulator, macos-arm64_x86_64, …
```

## 2. The two files into the example

```bash
EXP=~/REPOS/local-first-ai/tasks/chronos/exp_059_same_weights_three_chips
cp $EXP/kit/ios/KevBatch.swift $EXP/kit/ios/Exp059View.swift examples/llama.swiftui/llama.swiftui/
sed -i '' 's/ContentView()/Exp059View()/' examples/llama.swiftui/llama.swiftui/llama_swiftuiApp.swift
open examples/llama.swiftui/llama.swiftui.xcodeproj
```

In Xcode, add the two files to the target if they are not picked up: File → Add Files to "llama.swiftui"…, select both,
tick the `llama.swiftui` target.

## 3. Project settings (once)

- **Signing & Capabilities:** Team = your Apple ID (a Personal Team is enough; the app then expires after 7 days and is
  re-installed by running again). Bundle Identifier = something of yours, e.g. `eu.localfirstai.exp059`.
- **+ Capability → Increased Memory Limit** (`com.apple.developer.kernel.increased-memory-limit`). The Q8_0 file is
  4.48 GB; without this the per-app limit may refuse it.
- **Info tab → add two keys:** `Application supports iTunes file sharing` = YES (`UIFileSharingEnabled`) and
  `Supports opening documents in place` = YES (`LSSupportsOpeningDocumentsInPlace`). This makes the app's Documents
  folder visible in Finder and the Files app.

## 4. On the phone

- Settings → Privacy & Security → **Developer Mode** on (restart). Connect over USB, trust the Mac.
- Xcode: destination = the iPhone, **Run**. First launch: Settings → General → VPN & Device Management → trust the
  developer app.
- **Copy the inputs in:** Finder → iPhone → **Files** tab → drag `Kev-4B-Q8_0.gguf` (sha256 `7c2ebed9…`) and
  `$EXP/data/prompts59.jsonl` onto the app's folder. (Or AirDrop → Save to Files → On My iPhone → the app.)
- **Run conditions:** airplane mode, Wi-Fi off, Bluetooth off, screen brightness fixed, phone flat on a desk at room
  temperature, charger unplugged (so the battery figure means something). Tap **Run**. The log shows every 10
  enquiries; the loop is resumable if the app is killed.
- **Copy the outputs out:** Finder → iPhone → Files → the app → drag `raw_iphone.jsonl` and `run_iphone.json` to
  `$EXP/raw/` and rename them **`iphone.jsonl`** and **`iphone.run.json`**.

If the model fails to load with a memory error, stop: that is the declared fallback in HYPOTHESIS.md (a Q4_K_M build on
both arms, as a dated amendment), not something to work around in the app.

## 5. Control C2: the same app on the MacBook

- Xcode destination = **My Mac (Designed for iPad)**, Run. Its Documents folder is
  `~/Library/Containers/<bundle id>/Data/Documents/`; put the same two files there and tap Run.
- Copy `raw_iphone.jsonl` out as **`$EXP/raw/mbp-app.jsonl`** and `run_iphone.json` as `mbp-app.run.json`.

## 6. The MacBook server arm

```bash
cd ~/models/llama.cpp-b11711
cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j -t llama-server
./build/bin/llama-server -m ~/models/Kev-4B-Q8_0.gguf --host 127.0.0.1 --port 18442 -c 16384 -b 8192 -ub 8192
# in a second terminal
python3 $EXP/kit/run59_mac.py mbp
```

## 7. Finish and score (any Mac, no model needed)

```bash
cd $EXP
for arm in mini-a mini-b mbp iphone mbp-app; do [ -f raw/$arm.jsonl ] && python3 kit/finish59.py $arm; done
python3 kit/score59.py
```

## What the app does, in one paragraph

For each enquiry it tokenizes the 12 rendered prompts (special tokens parsed, no BOS), finds their shared prefix, decodes
that once, then decodes each question's remainder with outputs on, reads the 512-wide head at the last token and at each
`<|box_end|>`, scores every option as the dot product of the two halves over `sqrt(256)`, applies a softmax at the
temperature the model declares, and writes the server's answer shape. It records wall time, prefix time, token count,
thermal state, battery level, resident and available memory per enquiry, and the device identity, OS version, physical
memory, model hash and the metadata it read, once per run.
