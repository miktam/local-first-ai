#!/usr/bin/env python3
"""exp_057: drive one model through the frozen prompt set, recording each generation's wall-clock window and output.

Called by run57.py inside a watched window. Uses Ollama's /api/generate (stream, no tools), as the bot calls a
model; an embedding model (name starting `nomic-embed`) goes through /api/embed instead, as the bot's embedding
path does, and its "answer" records only the vector's length. Writes one JSON line per prompt with start and end
times (so an endpoint can be tied to a prompt) and the full answer (so score57.py can check for tool-call or
search-request structure, H4). Loaded with the audit hook, so the driver's own loopback call to 127.0.0.1:11434 is
in AUDIT57_LOG (control C_loop), and so would be any other network act of the driver's.

    python3.14 drive.py <model> <prompts.json> <out.jsonl> [max_prompts]

max_prompts exists for smoke tests only; the registered run sends every prompt.
"""
import json
import sys
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
EMBED_PREFIXES = ("nomic-embed",)


def is_embed(model):
    return model.startswith(EMBED_PREFIXES)


def _post(path, body, timeout):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def generate(model, prompt):
    body = {"model": model, "prompt": prompt, "stream": True, "options": {"temperature": 0.3, "num_ctx": 8192}}
    chunks = []
    with _post("/api/generate", body, 300) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("response"):
                chunks.append(obj["response"])
            if obj.get("done"):
                break
    return "".join(chunks)


def embed(model, prompt):
    with _post("/api/embed", {"model": model, "input": prompt}, 300) as resp:
        obj = json.loads(resp.read())
    vecs = obj.get("embeddings") or []
    return f"<embedding: {len(vecs)} vector(s) of {len(vecs[0]) if vecs else 0}>"


def main():
    model, prompts_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    prompts = json.loads(open(prompts_path).read())["prompts"]
    if len(sys.argv) > 4:
        prompts = prompts[:int(sys.argv[4])]
    call = embed if is_embed(model) else generate
    with open(out_path, "a", buffering=1, encoding="utf-8") as fh:
        for i, prompt in enumerate(prompts):
            t0 = time.time()
            try:
                answer, err = call(model, prompt), None
            except Exception as e:                      # noqa: BLE001 - record any failure, keep going
                answer, err = "", repr(e)
            t1 = time.time()
            fh.write(json.dumps({"model": model, "i": i, "prompt": prompt, "t0": t0, "t1": t1,
                                 "seconds": round(t1 - t0, 2), "error": err, "answer": answer}) + "\n")
            print(f"  [{model}] prompt {i}: {round(t1 - t0, 1)}s" + (f" ERROR {err}" if err else ""))


if __name__ == "__main__":
    main()
