// exp_059 — the phone arm's readout. Reproduces llama.cpp b11711's `/v1/systemone` answer for a `kev` decision model
// (tools/server/server-decision.cpp, server-context.cpp: send_decision) with the engine's public API only:
//   • the prompt is already rendered by kit/render59.py; it is tokenized with special tokens parsed, no BOS;
//   • the context runs with embeddings on and pooling off, so each output token yields the model's 512-wide head
//     (`cls.output`, applied by the Qwen3.5 graph), two halves of 256;
//   • an option's score = dot(first half at the last token, second half at the option's <|box_end|> token) / sqrt(256);
//   • softmax at the model's calibrated temperature (metadata <arch>.decision.temperature.<type>, read as float as the
//     server does); choice → argmax, probabilities, TypeSafe confidence; noul → the probability of `true`.
// The shared prefix of one enquiry's 12 prompts (<|fim_prefix|> state <|fim_middle|>) is decoded once, as the server
// shares it between the questions of one request. No token is ever generated, so there is no sampling and no seed.
//
// Drop this file and Exp059View.swift into examples/llama.swiftui/llama.swiftui/ of the b11711 checkout; see README.md.

import Foundation
import llama

struct Q59: Codable {
    let qid: String
    let type: String       // "choice" | "noul"
    let keys: [String]     // option keys in prompt order (noul: ["false", "true"])
    let prompt: String
}

struct Lead59: Codable {
    let id: String
    let lang: String?
    let questions: [Q59]
}

struct LeadResult {
    let answers: [String: Any]   // qid → the server's answer shape
    let msPrefix: Double
    let nPrefix: Int
    let nTokens: Int
}

enum KevError: Error, CustomStringConvertible {
    case load(String), layout(String), decode(String), readout(String)
    var description: String {
        switch self {
        case .load(let s):    return "load: \(s)"
        case .layout(let s):  return "layout: \(s)"
        case .decode(let s):  return "decode: \(s)"
        case .readout(let s): return "readout: \(s)"
        }
    }
}

// named apart from the example's llama_batch_add so both files can stay in the target
func kb_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq: llama_seq_id, _ output: Bool) {
    let i = Int(batch.n_tokens)
    batch.token[i] = id
    batch.pos[i] = pos
    batch.n_seq_id[i] = 1
    batch.seq_id[i]![0] = seq
    batch.logits[i] = output ? 1 : 0
    batch.n_tokens += 1
}

fileprivate func kbMeta(_ model: OpaquePointer, _ key: String) -> String? {
    var buf = [CChar](repeating: 0, count: 1024)
    let n = llama_model_meta_val_str(model, key, &buf, buf.count)
    return n < 0 ? nil : String(cString: buf)
}

/// common_tokenize(vocab, text, add_special = false, parse_special = true)
fileprivate func kbTokenize(_ vocab: OpaquePointer, _ text: String) -> [llama_token] {
    let n = Int32(text.utf8.count)
    var cap = Int(n) + 16
    var out = [llama_token](repeating: 0, count: cap)
    var r = llama_tokenize(vocab, text, n, &out, Int32(cap), false, true)
    if r < 0 {
        cap = Int(-r)
        out = [llama_token](repeating: 0, count: cap)
        r = llama_tokenize(vocab, text, n, &out, Int32(cap), false, true)
    }
    return r < 0 ? [] : Array(out[0..<Int(r)])
}

actor KevBatch {
    private let model: OpaquePointer
    private let ctx: OpaquePointer
    private let vocab: OpaquePointer
    private var batch: llama_batch
    private let nBatch: Int32
    let nCtx: UInt32
    let nEmbdOut: Int
    let marker: llama_token
    let arch: String
    let temperatures: [String: Double]   // as parsed from the metadata text
    let modelDesc: String

    init(modelPath: String, nCtx: UInt32 = 8192, nBatch: Int32 = 4096) throws {
        llama_backend_init()
        var mp = llama_model_default_params()
#if targetEnvironment(simulator)
        mp.n_gpu_layers = 0
#endif
        guard let m = llama_model_load_from_file(modelPath, mp) else {
            throw KevError.load("llama_model_load_from_file failed for \(modelPath)")
        }
        var cp = llama_context_default_params()
        cp.n_ctx = nCtx
        cp.n_batch = UInt32(nBatch)
        cp.n_ubatch = UInt32(nBatch)
        cp.n_seq_max = 1
        cp.embeddings = true                      // the server toggles this on for a decision batch
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE // per-token outputs; the gate resolves UNSPECIFIED to NONE for Kev
        let threads = Int32(max(1, min(8, ProcessInfo.processInfo.activeProcessorCount - 2)))
        cp.n_threads = threads
        cp.n_threads_batch = threads
        guard let c = llama_init_from_model(m, cp) else {
            llama_model_free(m)
            throw KevError.load("llama_init_from_model failed (n_ctx \(nCtx), n_batch \(nBatch))")
        }
        let v: OpaquePointer = llama_model_get_vocab(m)
        let toks = kbTokenize(v, "<|box_end|>")
        guard toks.count == 1 else {
            llama_free(c)
            llama_model_free(m)
            throw KevError.layout("<|box_end|> tokenizes to \(toks.count) tokens, expected 1")
        }
        let arch = kbMeta(m, "general.architecture") ?? ""
        var t: [String: Double] = [:]
        for k in ["choice", "score", "noul"] {
            if let s = kbMeta(m, "\(arch).decision.temperature.\(k)"), let d = Double(s) { t[k] = d }
        }
        var desc = [CChar](repeating: 0, count: 256)
        _ = llama_model_desc(m, &desc, desc.count)

        self.model = m
        self.ctx = c
        self.vocab = v
        self.nBatch = nBatch
        self.nCtx = nCtx
        self.batch = llama_batch_init(nBatch, 0, 1)
        self.nEmbdOut = Int(llama_model_n_embd_out(m))
        self.marker = toks[0]
        self.arch = arch
        self.temperatures = t
        self.modelDesc = String(cString: desc)
    }

    deinit {
        llama_batch_free(batch)
        llama_free(ctx)
        llama_model_free(model)
        llama_backend_free()
    }

    func info() -> [String: Any] {
        return ["arch": arch, "model_desc": modelDesc, "n_embd_out": nEmbdOut, "marker_token": Int(marker),
                "temperatures": temperatures, "n_ctx": Int(nCtx), "n_batch": Int(nBatch),
                "decision_type": kbMeta(model, "\(arch).decision.type") ?? "",
                "general_name": kbMeta(model, "general.name") ?? ""]
    }

    /// tokens[start..<end] at positions start..<end on sequence 0; with outputs, one llama_decode call (the readout
    /// indexes the outputs of that call)
    private func decode(_ toks: [llama_token], from start: Int, to end: Int, output: Bool) throws {
        if output && end - start > Int(nBatch) {
            throw KevError.layout("question remainder of \(end - start) tokens exceeds n_batch \(nBatch)")
        }
        var i = start
        while i < end {
            let stop = min(end, i + Int(nBatch))
            batch.n_tokens = 0
            for j in i..<stop {
                kb_add(&batch, toks[j], llama_pos(j), 0, output)
            }
            if llama_decode(ctx, batch) != 0 {
                throw KevError.decode("llama_decode failed at position \(i), \(stop - i) tokens")
            }
            i = stop
        }
    }

    func answer(_ lead: Lead59) throws -> LeadResult {
        let toks = lead.questions.map { kbTokenize(vocab, $0.prompt) }
        guard let minLen = toks.map({ $0.count }).min(), minLen > 1, let maxLen = toks.map({ $0.count }).max() else {
            throw KevError.layout("\(lead.id): empty prompt")
        }
        if maxLen > Int(nCtx) {
            throw KevError.layout("\(lead.id): a prompt has \(maxLen) tokens, n_ctx is \(nCtx)")
        }

        // the shared prefix: every question keeps at least one token of its own (server_decision_group_tasks),
        // and every marker stays in the remainder, where outputs are read
        var L = minLen - 1
        for t in toks.dropFirst() {
            var k = 0
            while k < L && t[k] == toks[0][k] { k += 1 }
            L = min(L, k)
        }
        for t in toks {
            if let m = t.firstIndex(of: marker) { L = min(L, m) }
        }

        let mem = llama_get_memory(ctx)
        llama_memory_clear(mem, true)
        let t0 = DispatchTime.now().uptimeNanoseconds
        if L > 0 {
            try decode(toks[0], from: 0, to: L, output: false)
        }
        let msPrefix = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6

        let np = nEmbdOut / 2
        var answers: [String: Any] = [:]
        var nTokens = L
        for (qi, q) in lead.questions.enumerated() {
            let t = toks[qi]
            if qi > 0 {
                _ = llama_memory_seq_rm(mem, 0, llama_pos(L), -1)   // drop the previous question's remainder
            }
            try decode(t, from: L, to: t.count, output: true)
            nTokens += t.count - L

            let markers = t.indices.filter { t[$0] == marker }
            guard markers.count == q.keys.count else {
                throw KevError.layout("\(lead.id)/\(q.qid): \(markers.count) markers for \(q.keys.count) options")
            }
            guard let eq = llama_get_embeddings_ith(ctx, Int32(t.count - 1 - L)) else {
                throw KevError.readout("\(lead.id)/\(q.qid): no embedding at the pointer")
            }
            var scores: [Float] = []
            scores.reserveCapacity(markers.count)
            for m in markers {
                guard let e = llama_get_embeddings_ith(ctx, Int32(m - L)) else {
                    throw KevError.readout("\(lead.id)/\(q.qid): no embedding at marker \(m)")
                }
                var dot: Float = 0
                for i in 0..<np {
                    dot += eq[i] * e[np + i]
                }
                scores.append(dot / sqrtf(Float(np)))
            }
            answers[q.qid] = format(q, scores)
        }
        return LeadResult(answers: answers, msPrefix: msPrefix, nPrefix: L, nTokens: nTokens)
    }

    /// server_decision_context::format_answer for one variant: softmax at the float temperature, then the fields
    private func format(_ q: Q59, _ scores: [Float]) -> [String: Any] {
        let temperature = Double(Float(temperatures[q.type] ?? 1.0))   // the server keeps it as a float
        let logits = scores.map { Double($0) }
        let mx = logits.max() ?? 0
        let e = logits.map { exp(($0 - mx) / temperature) }
        let s = e.reduce(0, +)
        let p = e.map { $0 / s }
        if q.type == "noul" {
            let i = q.keys.firstIndex(of: "true") ?? 1
            return ["type": "noul", "noul": p[i]]
        }
        var probs: [String: Double] = [:]
        for (k, v) in zip(q.keys, p) { probs[k] = v }
        var best = 0
        for i in 1..<p.count where p[i] > p[best] { best = i }    // first maximum, as std::max_element
        let uniform = 1.0 / Double(p.count)
        let confidence = p.count < 2 ? 1.0 : max(0.0, (p[best] - uniform) / (1.0 - uniform))
        return ["type": "choice", "choice": q.keys[best], "probabilities": probs, "confidence": confidence]
    }
}
