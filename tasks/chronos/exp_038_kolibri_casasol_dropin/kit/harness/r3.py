"""R3 (DESIGN §3.5): the exp_038 MLX driver reproduces exp_037's runner token for token. mbp only, real weights.

For each MLX arm (K4, K8): the first 5 step0 questions (S5, not an eval set) are rendered through the vendored glue,
with its retrieval, labels and options, harvested by a stub that returns IN for every gate. That gives 5 gate prompts
and 5 answer prompts. Each prompt is generated greedily three ways:
  (a) exp_037 runner.generate.run_cell, B = 1, the same max_tokens and EOS, greedy sampler;
  (b) kit/mlx_backend.py MLXGatedBackend.chat at repeat_penalty 1.0 (no processor installed);
  (c) the same, with a penalty processor at 1.0 attached at insert (spy), the processor path at identity.
Pass: (a), (b) and (c) give identical completion token ids on all 20 prompts per arm. Token ids are captured by a spy
on the generator; the backend itself is not changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import os

P38 = Path(__file__).resolve().parents[1]
VENDOR = P38 / "vendor" / "coapi_voice"
OUT = Path(os.environ.get("EXP038_R3_OUT", P38 / "runs"))  # tests write elsewhere


def harvest(n=5) -> list[dict]:
    sys.path[:0] = [str(VENDOR)]
    import glue as G  # noqa: E402
    import guide_bm25 as g  # noqa: E402
    import run_eval as ev  # noqa: E402

    rows = [json.loads(l) for l in (VENDOR / "data/step0_questions.jsonl").read_text().splitlines() if l.strip()][:n]
    idx = {lang: g.load_index(lang) for lang in g.LANGS}
    calls = []

    def stub(model, messages, options):
        calls.append({"messages": messages, "options": dict(options)})
        return {"message": {"content": "IN" if options.get("num_predict") == 4 else "stub."}}

    real = ev.ollama_chat
    ev.ollama_chat = stub
    try:
        for r in rows:
            G.answer(r["q"], model="r3", index=idx, verify=False)
    finally:
        ev.ollama_chat = real
    return calls


def main() -> int:
    sys.path.insert(0, str(P38 / "harness"))
    from mlx_entry import exact_fp32  # exp_037's ensure_exact_fp32, before any GPU work

    exact_fp32()
    sys.path[:0] = [str(P38 / "kit")]
    import mlx_backend as mb  # noqa: E402
    mb._e37_on_path()
    from mlx_lm.sample_utils import make_repetition_penalty  # noqa: E402
    from runner import generate, jsonl  # noqa: E402

    calls = harvest()
    out = {"check": "R3", "n_prompts": len(calls), "arms": {}}
    for arm in ("K4", "K8"):
        b = mb.MLXGatedBackend(arm, effort_answer="none")
        captured: list[list[int]] = []
        real_make = generate.make_batch_generator
        force_proc = {"on": False}

        def spy(*a, **k):
            g = real_make(*a, **k)
            ins, nxt = g.insert, g.next
            toks: list[int] = []
            captured.append(toks)

            def insert(prompts, **kw):
                if force_proc["on"]:
                    kw["logits_processors"] = [[make_repetition_penalty(1.0, 256)]]
                return ins(prompts, **kw)

            def next_():
                p, resps = nxt()
                toks.extend(int(r.token) for r in resps)
                return p, resps

            g.insert, g.next = insert, next_
            return g

        generate.make_batch_generator = spy
        rows = []
        try:
            for i, c in enumerate(calls):
                opts = dict(c["options"], repeat_penalty=1.0)
                captured.clear(); force_proc["on"] = False
                rb = b.chat(arm, c["messages"], opts)
                ids_b = list(captured[-1])
                captured.clear(); force_proc["on"] = True
                b.chat(arm, c["messages"], opts)
                ids_c = list(captured[-1])
                force_proc["on"] = False
                # (a) exp_037's run_cell on the same rendered prompt
                text, ids, rsha = b._render(b.family, b.tokenizer, c["messages"], "none")
                path = OUT / "r3_tmp" / f"{arm}_{i}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                w = jsonl.open_run(path, {"cell": "r3", "B": 1}, aborted_root=path.parent / "aborted", private_root=path.parent / "private")
                try:
                    generate.make_batch_generator = real_make
                    generate.run_cell(b.model, b.tokenizer, [{"id": f"r3-{i}", "item_sha256": "0" * 64, "prompt_sha256": rsha,
                                                              "prompt_ids": ids, "prompt_text": text}],
                                      arm, "r3", "none", rb["exp038"]["max_tokens"], 1, 0, w, b.family,
                                      sampler_cfg={"order": "vllm", "temperature": 0.0, "top_p": 1.0, "top_k": 0}, eos=b.eos)
                finally:
                    w.close()
                    generate.make_batch_generator = spy
                rec = list(jsonl.first_records(jsonl.read_jsonl(path)))[0]
                ids_a = rec["completion_ids"]
                rows.append({"i": i, "call": rb["exp038"]["call"], "n": len(ids_a), "a_eq_b": ids_a == ids_b, "a_eq_c": ids_a == ids_c})
        finally:
            generate.make_batch_generator = real_make
        out["arms"][arm] = {"rows": rows, "pass": all(r["a_eq_b"] and r["a_eq_c"] for r in rows), "binding": b.binding}
        del b
    out["verdict"] = "PASS" if all(v["pass"] for v in out["arms"].values()) else "FAIL"
    dst = OUT / "r3_result.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({a: v["pass"] for a, v in out["arms"].items()} | {"verdict": out["verdict"]}))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
