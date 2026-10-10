"""exp_038 MLXGatedBackend: Kolibri through exp_037's gated runner, behind the A12 glue's model calls.

The glue calls `run_eval.ollama_chat(model, messages, options)` for its scope gate (num_predict 4) and its
answer (run_eval.OPTIONS). The harness points that name at `MLXGatedBackend.chat`, which returns the same
shape as Ollama's /api/chat reply. The glue itself is never edited.

What this module takes from exp_037 (imported read-only from E37, bound by tree hash, never copied):
- runner.chat.render: the frozen Kolibri template, with an explicit `reasoning_effort`;
- runner.generate.make_batch_generator: the kit's only BatchGenerator construction (fp32 logits, EOS stops);
- runner.generate._settle_batch_offsets: Amendment 2's per-step settle of the BatchKVCache offsets;
- runner.generate._scorer_split: Amendment 1's uncached tokenizer check before the scorers' reasoning split;
- runner.sampler.make_vllm_sampler at temperature 0: argmax;
- runner.run.Loaded: port-file check, sampling and template checks, EOS ids, trusted load;
- runner.guard: gate, environment and record checks (rev 2 §0b.1).

What is exp_038's own (DESIGN rev 2, §0b.2-3, §2.1; P38 runs/w0r_sampler_20261010.json):
- one fresh batch generator per call, at B = 1;
- the answer call's repeat penalty as G26 applies it on Ollama 0.40.2 (llamacpp runner): 1.05 over the last
  256 tokens of prompt + completion, through mlx-lm's own `make_repetition_penalty`, whose rule equals
  llama.cpp's penalties sampler at presence 0 and frequency 0. The gate call carries no penalty (Ollama's
  default is 1.0, and gemma4:26b's Modelfile sets none);
- K4-MED: only the answer call reasons, at `reasoning_effort="medium"`, under a reasoning cap. The visible
  answer keeps the glue's num_predict. The gate call always renders at effort "none".
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

E37 = Path(os.environ.get("EXP038_E37", Path.home() / "REPOS/local-first-ai/tasks/chronos/exp_037_kolibri_forced_gate"))

# The record that passed (rev 2 §0b.1).
GATE_RECORD = "gate_20261008T180634Z.json"
GATE_RECORD_SHA256 = "e6d80d7b4565fef08ba275b1c15f635880c3155441784c713ff4d8b296c62341"
RUNNER_SHA256 = "75a50caa8d541396c7a5f82468b50049330cded96426255ecc56deff1ee83c87"
PORT_SHA256_PREFIX = "2c153357"

GATE_NUM_PREDICT = 4  # the glue's scope gate (glue.py in_scope)


class BindingError(SystemExit):
    pass


def _e37_on_path() -> None:
    for p in (str(E37), str(E37 / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runner_tree_sha256() -> str:
    _e37_on_path()
    import hash_tree  # E37 tools/hash_tree.py

    scope = next(s for s in hash_tree.SCOPES if s.placeholder == "RUNNER_SHA256")
    return hash_tree.scope_value(scope, root=E37)


def bind(arm: str) -> dict:
    """§2.2, real weights only. Refuses unless exp_037's newest real gate record is run 3's, exit 0, the
    environment matches its pins, the runner tree is the post-Amendment-2 tree and the port file is exp_037's."""
    _e37_on_path()
    from runner import guard

    if arm not in ("K4", "K8"):
        raise BindingError(f"exp_038 binds Kolibri arms only, not {arm!r}")
    guard.require_gate(arm)
    newest = guard.newest_gate_record()  # a dict with "_file" (runner/guard.py)
    name = (newest or {}).get("_file")
    if name != GATE_RECORD:
        raise BindingError(f"newest gate record is {name}, exp_038 is bound to {GATE_RECORD}")
    got = _sha256_file(E37 / "results" / "gate" / GATE_RECORD)
    if got != GATE_RECORD_SHA256:
        raise BindingError(f"{GATE_RECORD} sha256 {got[:12]} != {GATE_RECORD_SHA256[:12]}")
    rec = json.loads((E37 / "results" / "gate" / GATE_RECORD).read_text(encoding="utf-8"))
    if rec.get("mode") != "real" or rec.get("dry_run_promoted_from"):
        raise BindingError(f"{GATE_RECORD} is not a real-mode record")
    if rec.get("verdict") != {"K8": "PASS", "K4": "PASS"}:
        raise BindingError(f"{GATE_RECORD} verdict {rec.get('verdict')} is not exit 0")
    guard.require_environment(rec)
    tree = runner_tree_sha256()
    if tree != RUNNER_SHA256:
        raise BindingError(f"E37 runner tree {tree[:12]} != {RUNNER_SHA256[:12]} (post-Amendment-2)")
    return {"gate_record": GATE_RECORD, "gate_record_sha256": GATE_RECORD_SHA256, "runner_sha256": tree}


class MLXGatedBackend:
    """One Kolibri arm. `chat()` has run_eval.ollama_chat's signature and reply shape.

    Real weights: MLXGatedBackend("K4", effort_answer=...) binds (§2.2) and loads through E37's Loaded.
    Tests: MLXGatedBackend.tiny(model, tokenizer, eos) skips the binding and marks every reply "tiny"."""

    def __init__(self, arm: str, effort_answer: str = "none", reasoning_cap: int | None = None, *,
                 _tiny: tuple | None = None):
        _e37_on_path()
        from runner import chat, generate
        from runner.sampler import make_vllm_sampler

        if effort_answer not in ("none", "medium"):
            raise ValueError(f"exp_038 runs Kolibri at effort none or medium, not {effort_answer!r}")
        if effort_answer != "none" and not reasoning_cap:
            raise ValueError("effort medium needs a reasoning cap (R1's pilot rule)")
        self.arm = arm
        self.effort_answer = effort_answer
        self.reasoning_cap = int(reasoning_cap or 0)
        self._chat, self._gen = chat, generate
        self.sampler = make_vllm_sampler(0.0, 1.0, 0)
        if _tiny is not None:
            self.model, self.tokenizer, self.eos = _tiny
            self.family = "kolibri"
            self.binding: dict[str, Any] = {"tiny": True}
            self.meta: dict[str, Any] = {}
        else:
            self.binding = bind(arm)
            from runner.run import Loaded

            self.model, self.tokenizer, self.meta = Loaded().get(arm)
            port = str(self.meta.get("port_sha256", ""))
            if not port.startswith(PORT_SHA256_PREFIX):
                raise BindingError(f"port file {port[:12]} is not exp_037's {PORT_SHA256_PREFIX}…")
            self.binding["port_sha256"] = port
            self.family = chat.family_of(arm)
            self.eos = [int(e) for e in self.meta["eos_ids"]]
        self.open_id, self.close_id = generate.special_ids(self.tokenizer, self.family)
        self.scorer_split = generate._scorer_split(self.family, self.tokenizer)
        self._blank: dict[int, bool] = {}
        self._render = chat.render  # tests on tiny builds replace it (no Kolibri tokenizer there)
        self._on_step = None  # tests: called with the generator after every settle

    @classmethod
    def tiny(cls, model, tokenizer, eos, effort_answer: str = "none", reasoning_cap: int | None = None):
        return cls("K4", effort_answer, reasoning_cap, _tiny=(model, tokenizer, [int(e) for e in eos]))

    # ------------------------------------------------------------------------------------------- helpers
    def _is_blank(self, t: int) -> bool:
        if t not in self._blank:
            self._blank[t] = self._gen._decode(self.tokenizer, [t]).strip() == ""
        return self._blank[t]

    def _decode_plain(self, ids: list[int]) -> str:
        from runner.chat import hf_tokenizer

        return hf_tokenizer(self.tokenizer).decode(ids, skip_special_tokens=True)

    def _reasoning_state(self, toks: list[int]) -> tuple[bool, int, int]:
        """(closed, reasoning tokens so far, answer tokens so far) while generating at effort medium."""
        j = 0
        while j < len(toks) and self._is_blank(toks[j]):
            j += 1
        if self.open_id is None or j >= len(toks) or toks[j] != self.open_id:
            return True, 0, len(toks)  # the model did not open a reasoning segment
        rest = toks[j + 1:]
        if self.close_id is not None and self.close_id in rest:
            k = rest.index(self.close_id)
            return True, k, len(rest) - k - 1
        return False, len(rest), 0

    # ---------------------------------------------------------------------------------------- the call
    def chat(self, model_name: str, messages: list[dict], options: dict) -> dict:
        call = "gate" if int(options.get("num_predict", 0)) == GATE_NUM_PREDICT else "answer"
        if float(options.get("temperature", 0.0)) != 0.0:
            raise ValueError("exp_038's MLX arms decode greedily; temperature must be 0")
        for k in ("presence_penalty", "frequency_penalty"):
            if float(options.get(k, 0.0)) != 0.0:
                raise ValueError(f"{k} != 0 is not implemented (G26's calls send 0)")
        effort = self.effort_answer if call == "answer" else "none"
        text, ids, rsha = self._render(self.family, self.tokenizer, messages, effort)
        n_ans = int(options["num_predict"])
        r_cap = self.reasoning_cap if effort != "none" else 0
        max_tokens = n_ans + (r_cap + 2 if r_cap else 0)  # + open and close delimiters
        rp = float(options.get("repeat_penalty", 1.0)) if call == "answer" else 1.0
        last_n = int(options.get("repeat_last_n", 64))
        procs = []
        if rp != 1.0:
            from mlx_lm.sample_utils import make_repetition_penalty

            procs = [make_repetition_penalty(rp, last_n)]

        t0 = time.perf_counter()
        gen = self._gen.make_batch_generator(self.model, 1, self.eos, max_tokens, self.sampler)
        toks: list[int] = []
        finish = None
        steps = 0
        try:
            gen.insert([ids], max_tokens=[max_tokens], logits_processors=[procs] if procs else None)
            while finish is None:
                _, resps = gen.next()
                self._gen._settle_batch_offsets(gen)  # Amendment 2, every step
                if self._on_step is not None:
                    self._on_step(gen)
                steps += 1
                for r in resps:
                    toks.append(int(r.token))
                    if r.finish_reason is not None:
                        finish = r.finish_reason
                if finish is None and r_cap:
                    closed, n_r, n_a = self._reasoning_state(toks)
                    if not closed and n_r >= r_cap:
                        finish = "reasoning_cap"
                    elif closed and n_a >= n_ans:
                        finish = "length"
                if steps > max_tokens + 8:
                    raise RuntimeError(f"no finish after {steps} steps (max_tokens {max_tokens})")
        finally:
            gen.close()
        wall = time.perf_counter() - t0

        if self.scorer_split is not None:
            r_ids, a_ids, status = self.scorer_split(ids, toks, self.eos, self._is_blank)
            split_by = "scorers.reasoning.split_ids"
        else:
            prefill = self._chat.prompt_prefill(self.family, text)
            r_ids, a_ids, status = self._gen.split_ids(self.family, prefill, toks, self.open_id, self.close_id,
                                                      self.eos, self._is_blank)
            split_by = "runner.generate.split_ids"
        if finish == "reasoning_cap":
            a_ids = []  # no answer exists; the row is scored as it stands (R1's rule)
        content = self._decode_plain(a_ids)
        return {
            "model": model_name,
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "length" if finish in ("length", "reasoning_cap") else "stop",
            "prompt_eval_count": len(ids),
            "eval_count": len(a_ids),
            "exp038": {
                "arm": self.arm, "call": call, "effort": effort, "rendered_sha256": rsha,
                "prompt_tokens": len(ids), "num_ctx": options.get("num_ctx"),
                "prompt_over_num_ctx": bool(options.get("num_ctx")) and len(ids) > int(options["num_ctx"]),
                "completion_tokens": len(toks), "reasoning_tokens": len(r_ids), "answer_tokens": len(a_ids),
                "reasoning_status": status, "split_by": split_by, "finish_reason": finish,
                "reasoning_cap": r_cap, "max_tokens": max_tokens, "decode_calls": steps,
                "repeat_penalty": rp, "repeat_last_n": last_n if rp != 1.0 else None,
                "wall_s": round(wall, 3), "binding": dict(self.binding),
            },
        }
