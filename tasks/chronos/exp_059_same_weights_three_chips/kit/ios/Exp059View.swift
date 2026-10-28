// exp_059 — the phone arm's screen and run loop. Reads Documents/<model>.gguf and Documents/prompts59.jsonl, writes
// Documents/raw_iphone.jsonl (one line per enquiry, resumable) and Documents/run_iphone.json (device, hashes, settings).
// With UIFileSharingEnabled the Documents folder is visible in Finder and the Files app, so files move over USB.

import SwiftUI
import CryptoKit
import Darwin
import os

private func machineModel() -> String {
    var s = utsname()
    uname(&s)
    return withUnsafePointer(to: &s.machine) {
        $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) }
    }
}

private func residentMB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.resident_size) / 1048576.0 : -1
}

private func thermalName(_ s: ProcessInfo.ThermalState) -> String {
    switch s {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

private func sha256(of url: URL, progress: (Int64) -> Void) throws -> String {
    let h = try FileHandle(forReadingFrom: url)
    defer { try? h.close() }
    var hasher = SHA256()
    var total: Int64 = 0
    while true {
        let more: Bool = autoreleasepool {
            let d = h.readData(ofLength: 16 << 20)
            if d.isEmpty { return false }
            hasher.update(data: d)
            total += Int64(d.count)
            progress(total)
            return true
        }
        if !more { break }
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

private func iso(_ d: Date = Date()) -> String {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return f.string(from: d)
}

/// the batch, off the main actor
final class Exp059Runner: @unchecked Sendable {
    var stop = false

    func run(docs: URL, log: @escaping @Sendable (String) -> Void) async {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil) else {
            log("cannot list Documents"); return
        }
        let ggufs = files.filter { $0.pathExtension == "gguf" }
        guard let model = ggufs.first(where: { $0.lastPathComponent == "Kev-4B-Q8_0.gguf" }) ?? ggufs.first else {
            log("no .gguf in Documents — copy Kev-4B-Q8_0.gguf in with Finder (iPhone → Files) or the Files app"); return
        }
        let promptsURL = docs.appendingPathComponent("prompts59.jsonl")
        guard let promptsData = fm.contents(atPath: promptsURL.path), let promptsText = String(data: promptsData, encoding: .utf8) else {
            log("no prompts59.jsonl in Documents"); return
        }
        let dec = JSONDecoder()
        var leads: [Lead59] = []
        for line in promptsText.split(separator: "\n") where !line.isEmpty {
            guard let l = try? dec.decode(Lead59.self, from: Data(line.utf8)) else { log("bad line in prompts59.jsonl"); return }
            leads.append(l)
        }
        let promptsSha = SHA256.hash(data: promptsData).map { String(format: "%02x", $0) }.joined()
        log("\(leads.count) enquiries, prompts sha256 \(promptsSha.prefix(12))")

        let rawURL = docs.appendingPathComponent("raw_iphone.jsonl")
        var done = Set<String>()
        if let t = fm.contents(atPath: rawURL.path).flatMap({ String(data: $0, encoding: .utf8) }) {
            for line in t.split(separator: "\n") {
                if let o = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any], let id = o["id"] as? String { done.insert(id) }
            }
        }
        if !done.isEmpty { log("resuming: \(done.count) already done") }

        log("hashing \(model.lastPathComponent)…")
        var lastGB: Int64 = 0
        let modelSha: String
        do {
            modelSha = try sha256(of: model) { n in
                if n >> 30 > lastGB { lastGB = n >> 30; log("  \(lastGB) GB") }
            }
        } catch { log("hash failed: \(error)"); return }
        log("model sha256 \(modelSha.prefix(12))")

        let sysVersion = await MainActor.run { UIDevice.current.systemVersion }
        let pi = ProcessInfo.processInfo
        var runInfo: [String: Any] = [
            "arm": "iphone", "experiment": "exp_059", "engine": "llama.cpp b11711 (9d1e036) via llama.xcframework",
            "device_model": machineModel(), "os_version": sysVersion, "physical_memory_mb": Double(pi.physicalMemory) / 1048576.0,
            "available_memory_mb_at_start": Double(os_proc_available_memory()) / 1048576.0,
            "active_processors": pi.activeProcessorCount, "model_file": model.lastPathComponent, "model_sha256": modelSha,
            "prompts_sha256": promptsSha, "started": iso(), "resumed_from": done.count,
        ]

        log("loading model…")
        let kev: KevBatch
        do { kev = try KevBatch(modelPath: model.path) } catch { log("load failed: \(error)"); return }
        let info = await kev.info()
        for (k, v) in info { runInfo[k] = v }
        runInfo["resident_mb_after_load"] = residentMB()
        runInfo["available_memory_mb_after_load"] = Double(os_proc_available_memory()) / 1048576.0
        log("loaded: \(info["model_desc"] ?? "") n_embd_out \(info["n_embd_out"] ?? 0) temperatures \(info["temperatures"] ?? [:]) resident \(Int(residentMB())) MB")
        writeJSON(runInfo, to: docs.appendingPathComponent("run_iphone.json"))

        if !fm.fileExists(atPath: rawURL.path) { fm.createFile(atPath: rawURL.path, contents: nil) }
        guard let out = try? FileHandle(forWritingTo: rawURL) else { log("cannot open raw_iphone.jsonl"); return }
        defer { try? out.close() }
        _ = try? out.seekToEnd()

        var n = done.count
        var errors = 0
        let tRun = DispatchTime.now().uptimeNanoseconds
        for lead in leads where !done.contains(lead.id) {
            if stop { log("stopped"); break }
            let battery = await MainActor.run { UIDevice.current.batteryLevel }
            let thermal = thermalName(pi.thermalState)
            let t0 = DispatchTime.now().uptimeNanoseconds
            do {
                let r = try await kev.answer(lead)
                let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
                let row: [String: Any] = [
                    "id": lead.id, "answers": r.answers, "ms": (ms * 10).rounded() / 10, "ms_prefix": (r.msPrefix * 10).rounded() / 10,
                    "n_prefix": r.nPrefix, "n_tokens": r.nTokens, "thermal": thermal, "battery": Double(battery),
                    "resident_mb": (residentMB()).rounded(), "available_mb": (Double(os_proc_available_memory()) / 1048576.0).rounded(),
                    "t": iso(),
                ]
                let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
                out.write(data)
                out.write(Data("\n".utf8))
                n += 1
                errors = 0
                if n % 10 == 0 || n == leads.count {
                    let el = Double(DispatchTime.now().uptimeNanoseconds - tRun) / 1e9
                    log("\(n)/\(leads.count)  \(Int(ms)) ms  \(r.nTokens) tok  \(thermal)  \(Int(el)) s elapsed")
                }
            } catch {
                errors += 1
                log("\(lead.id): \(error)")
                if errors >= 3 { log("three errors in a row, stopping"); break }
            }
        }
        runInfo["finished"] = iso()
        runInfo["n_done"] = n
        runInfo["resident_mb_at_end"] = residentMB()
        writeJSON(runInfo, to: docs.appendingPathComponent("run_iphone.json"))
        log("done: \(n)/\(leads.count) → raw_iphone.jsonl, run_iphone.json")
    }

    private func writeJSON(_ o: [String: Any], to url: URL) {
        if let d = try? JSONSerialization.data(withJSONObject: o, options: [.sortedKeys, .prettyPrinted]) {
            try? d.write(to: url)
        }
    }
}

@MainActor
final class Exp059Model: ObservableObject {
    @Published var log = ""
    @Published var running = false
    private var runner: Exp059Runner?

    func start() {
        guard !running else { return }
        running = true
        UIApplication.shared.isIdleTimerDisabled = true
        UIDevice.current.isBatteryMonitoringEnabled = true
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let r = Exp059Runner()
        runner = r
        log += "Documents: \(docs.path)\n"
        Task.detached(priority: .userInitiated) { [weak self] in
            await r.run(docs: docs) { msg in
                Task { @MainActor in self?.log += msg + "\n" }
            }
            await MainActor.run {
                self?.running = false
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }

    func stop() { runner?.stop = true }
}

struct Exp059View: View {
    @StateObject private var m = Exp059Model()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("exp_059 — same weights, three chips").font(.headline)
            Text("Documents: Kev-4B-Q8_0.gguf + prompts59.jsonl → raw_iphone.jsonl, run_iphone.json. Airplane mode, Wi-Fi off, screen on.")
                .font(.caption)
            ScrollViewReader { proxy in
                ScrollView {
                    Text(m.log)
                        .font(.system(size: 11, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .id("end")
                }
                .onChange(of: m.log) { _ in proxy.scrollTo("end", anchor: .bottom) }
            }
            HStack {
                Button(m.running ? "Running…" : "Run") { m.start() }.disabled(m.running)
                Button("Stop") { m.stop() }.disabled(!m.running)
            }
            .buttonStyle(.bordered)
        }
        .padding()
    }
}
