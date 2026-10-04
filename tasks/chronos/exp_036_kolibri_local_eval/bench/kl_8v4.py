# SPDX-License-Identifier: MIT
"""H8 — 4-bit logit fidelity against the peers (HYPOTHESIS H8; BUILD_SPEC §5.7
kl_8v4.py). Output: results/kl_8v4_<UTC>.json.

Texts: the decoded text of gate texts T1-T4 (gate/texts/T<n>_*.txt, committed)
and T5-T6 ($EXP036_WORK/gate_texts/T<n>.txt, never committed), each checked
against the text_sha256 that gate/texts/MANIFEST.json records for it.

Per model family (Kolibri, Gemma 4, Qwen3.6, Qwen3.8), per text:
- the family's own tokenizer (raw `tokenizers`, tokenizer.json of the 8-bit
  directory; the 4-bit directory must give the same ids) tokenises the text
  alone, with no special tokens; the token offsets give each token's first
  byte;
- chat-wrapped (Amendment 5, Andrei 2026-10-04): every model reads the text
  as the ASSISTANT TURN after one fixed user message, "Write a text.",
  rendered through its own chat template by runner.chat.render (the scored
  runs' rendering) with add_generation_prompt=True and thinking off
  (runner.chat.template_kwargs(family, "none"): Kolibri
  reasoning_effort="none" with its tokenizer_config template; Gemma 4,
  Qwen3.6 and Qwen3.8 enable_thinking=False with runner/templates/). The
  model teacher-forces prompt_ids + ids[:-1]; only the rows that predict the
  text's tokens are scored (score_from = len(prompt_ids) - 1), so every text
  token is predicted and no prompt token is. No extra BOS: the template
  emits any (Gemma 4 <bos>). The 8-bit and 4-bit directories must render
  the same prompt ids. chat_wrapper() records the prompt, kwargs and
  template sha256. Why: Gemma 4 IT teacher-forced on raw text after <bos>
  falls into its chat/thinking-channel format (about 11 nats/token on T1),
  which made its NLL and KL meaningless (aborted/20261004T143051Z-peercheck);
- the wrapper/text boundary (review of Amendment 5): because the text is
  tokenised alone, a text that starts with a newline begins with its own
  newline token. Kolibri's, Qwen3.6's and Qwen3.8's wrappers end on the
  token "\\n\\n" (Kolibri 263, Qwen 271), so on T1, T2 ("\\n") and T6
  ("\\n\\n") they score a split that their tokenizers never produce from the
  joint string (Kolibri ['\\n\\n', '\\n'] where the joint encoding gives
  '\\n\\n\\n' 120038; Qwen 1358; T6 '\\n\\n\\n\\n' 120724 / 987). Gemma 4's wrapper
  ends on the special token <channel|>, so nothing merges there. The scored
  ids stay as decided; each text records wrapper_boundary (wrapper_boundary())
  and the per-position stats of its first two scored rows (first_rows), so a
  sensitivity without those rows can be computed from the record;
- (context_prefix() and score_from = 0 give the raw-text measurement, which
  tools/peer_check.py still records for the gate's G3);
- logits come from chunked forwards through the model's own cache
  (chunk 2048) and are rounded to bf16 and upcast to fp32 for every model
  (like for like, C6); Kolibri's KL is also computed from its fp32 logits;
- per scored position KL(p8 || p4) = sum_v p8 (log p8 - log p4) in fp32,
  plus NLL of the true next token under each model and top-1 agreement;
- each text is cut into 8 byte-aligned blocks at whitespace (over the text's
  own UTF-8 bytes; the wrapper adds none); a token belongs to the block
  holding its first byte; per block the sums of KL, NLL and agreement, the
  token count and the byte count are written.

Kolibri: K8 is loaded alone, its log-probs (both variants, fp32 .npy) are
dumped to $EXP036_WORK/kl/<UTC>/, K8 is unloaded, K4 is loaded and compared
against the dump. Peers: the 8-bit and 4-bit models are loaded together.

The stratified block bootstrap, the ratio to the peer median and the H8 rule
are analysis/'s (stats.boot_blocks_stratified, verdicts.py); this cell writes
only sums.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import common, genutil  # noqa: E402

SPEC = "HYPOTHESIS H8; BUILD_SPEC §5.7 kl_8v4.py"
TEXT_IDS = ("T1", "T2", "T3", "T4", "T5", "T6")
PUBLIC_TEXTS = ("T1", "T2", "T3", "T4")
N_BLOCKS = 8
CHUNK = 2048  # teacher-forcing chunk, as teacher_force_logprobs (BUILD_SPEC §5.4)
ROW_SLICE = 256  # rows per KL evaluation, bounds the [rows, V] temporaries
WHITESPACE = b" \t\n\r"
FAMILIES = ("kolibri", "gemma4", "qwen3_6", "qwen3_8")
STATS = ("kl", "nll8", "nll4", "agree")
FIRST_ROWS = 2  # per-position stats of the first scored rows, recorded per text (the wrapper/text boundary)

# Amendment 5 (Andrei, 2026-10-04, "Chat-wrapped"): the fixed user turn every model answers with the gate text,
# and the effort that turns thinking off through runner.chat.template_kwargs.
WRAP_USER_MESSAGE = "Write a text."
WRAP_EFFORT = "none"
WRAP_AMENDMENT = "Amendment 5"
WRAP_RULE = ("the gate text is the assistant turn after the single user message 'Write a text.', rendered with "
             "runner.chat.render (the model's own chat template, add_generation_prompt=True, thinking off); "
             "input = prompt_ids + text_ids[:-1], text tokenised alone (add_special_tokens=False), no extra BOS; "
             "only the rows predicting the text's tokens are scored")


# ---------------------------------------------------------------------------
# Texts, blocks, token first bytes
# ---------------------------------------------------------------------------


def gate_text_paths(public_dir: Optional[Path] = None, work_texts_dir: Optional[Path] = None) -> dict[str, Path]:
    """T1-T4 from gate/texts/, T5-T6 from $EXP036_WORK/gate_texts/; exactly
    one T<n>_*.txt (or T<n>.txt) per text, else a hard failure."""
    pub = Path(public_dir) if public_dir else common.EXP_DIR / "gate" / "texts"
    work = Path(work_texts_dir) if work_texts_dir else common.work_dir() / "gate_texts"
    out = {}
    for t in TEXT_IDS:
        d = pub if t in PUBLIC_TEXTS else work
        found = sorted(d.glob(f"{t}_*.txt")) + ([d / f"{t}.txt"] if (d / f"{t}.txt").is_file() else [])
        if len(found) != 1:
            raise common.BenchError(f"gate text {t}: expected one {t}_*.txt in {common.redact_path(d)}, found {len(found)}")
        out[t] = found[0]
    return out


def load_texts(paths: dict[str, Path]) -> dict[str, dict]:
    texts = {}
    for t, p in paths.items():
        raw = p.read_bytes()
        text = raw.decode("utf-8")  # strict: invalid UTF-8 is a hard failure
        if "�" in text:
            raise common.BenchError(f"gate text {t} contains U+FFFD")
        texts[t] = {"text": text, "bytes": len(raw), "sha256": common.sha256_bytes(raw), "file": common.redact_path(p)}
    return texts


def manifest_text_sha256(public_dir: Path) -> dict[str, str]:
    """text_sha256 of T1-T6 as gate/texts/MANIFEST.json records them
    ("texts" for T1-T4, "web" for T5-T6); {} when there is no manifest."""
    m = Path(public_dir) / "MANIFEST.json"
    if not m.is_file():
        return {}
    obj = json.loads(m.read_text())
    out = {}
    for section in ("texts", "web"):
        for t, rec in (obj.get(section) or {}).items():
            if t in TEXT_IDS and isinstance(rec, dict) and rec.get("text_sha256"):
                out[t] = rec["text_sha256"]
    return out


def verify_texts(texts: dict[str, dict], expected: dict[str, str]) -> dict[str, Optional[bool]]:
    """Every text the manifest names must hash to its recorded sha256 (the
    H8 texts are the frozen gate texts); a mismatch is a hard failure."""
    out = {}
    for t, rec in texts.items():
        if t in expected:
            if rec["sha256"] != expected[t]:
                raise common.BenchError(f"gate text {t}: sha256 {rec['sha256']} != gate/texts/MANIFEST.json {expected[t]}")
            out[t] = True
        else:
            out[t] = None
    return out


def byte_blocks(data: bytes, n_blocks: int = N_BLOCKS) -> list[int]:
    """Block boundaries [0, b1, ..., b_{n-1}, len(data)].

    b_k is the first whitespace byte at or after round(k * len / n) that lies
    after b_{k-1}; failing that, the last whitespace byte between b_{k-1} and
    the target; failing that, the first UTF-8 character boundary at or after
    the target. The whitespace byte starts block k."""
    n = len(data)
    if n < n_blocks:
        raise common.BenchError(f"text of {n} bytes cannot be cut into {n_blocks} blocks")
    bounds = [0]
    for k in range(1, n_blocks):
        target = max(round(k * n / n_blocks), bounds[-1] + 1)
        b = next((i for i in range(target, n) if data[i] in WHITESPACE), None)
        if b is None:
            b = next((i for i in range(target - 1, bounds[-1], -1) if data[i] in WHITESPACE), None)
        if b is None:
            b = target
            while b < n and (data[b] & 0xC0) == 0x80:
                b += 1
        if not bounds[-1] < b < n:
            raise common.BenchError("could not place a block boundary")
        bounds.append(b)
    bounds.append(n)
    return bounds


def first_bytes(text: str, offsets: list[tuple[int, int]]) -> np.ndarray:
    """UTF-8 byte position of each token's first character, from the
    tokenizer's character offsets. A byte-level token that splits a
    multi-byte character carries that character's offsets, so it maps to the
    character's first byte. A zero-width span takes the previous token's."""
    cum = np.zeros(len(text) + 1, dtype=np.int64)
    cum[1:] = np.cumsum([len(c.encode("utf-8")) for c in text])
    out = np.empty(len(offsets), dtype=np.int64)
    prev = 0
    for j, (s, e) in enumerate(offsets):
        fb = int(cum[s]) if e > s else prev
        if fb < prev:
            raise common.BenchError("token offsets go backwards")
        out[j] = prev = fb
    return out


def token_blocks(fb: np.ndarray, bounds: list[int]) -> np.ndarray:
    """Block index of each token (the block holding its first byte)."""
    return np.searchsorted(np.asarray(bounds[1:-1]), fb, side="right")


def tokenize(tokenizer_json: Path, text: str) -> tuple[list[int], np.ndarray]:
    from tokenizers import Tokenizer

    enc = Tokenizer.from_file(str(tokenizer_json)).encode(text, add_special_tokens=False)
    return [int(i) for i in enc.ids], first_bytes(text, enc.offsets)


def context_prefix(model_dir: Path) -> dict:
    """The raw-text context-start token: the tokenizer's BOS if it has one,
    else generation_config.json bos_token_id, else <|endoftext|> (Gemma 4
    <bos>; Qwen generation_config bos = <|endoftext|>; Kolibri <|endoftext|>).
    Used only by the raw-text measurement (tools/peer_check.py, for G3)."""
    from tokenizers import Tokenizer

    d = Path(model_dir)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    tc = json.loads((d / "tokenizer_config.json").read_text()) if (d / "tokenizer_config.json").is_file() else {}
    gc = json.loads((d / "generation_config.json").read_text()) if (d / "generation_config.json").is_file() else {}
    bos = tc.get("bos_token")
    if isinstance(bos, dict):
        bos = bos.get("content")
    if bos:
        i = tok.token_to_id(bos)
        if i is not None:
            return {"id": int(i), "token": bos, "rule": "tokenizer_config bos_token"}
    if isinstance(gc.get("bos_token_id"), int):
        i = int(gc["bos_token_id"])
        return {"id": i, "token": tok.id_to_token(i), "rule": "generation_config bos_token_id"}
    i = tok.token_to_id("<|endoftext|>")
    if i is not None:
        return {"id": int(i), "token": "<|endoftext|>", "rule": "<|endoftext|> (no BOS defined)"}
    raise common.BenchError(f"no context-start token for {common.redact_path(d)}")


def load_chat_tokenizer(model_dir: Path):
    """The tokenizer the runner renders prompts with: mlx_lm's TokenizerWrapper
    (what mlx_lm.load returns), unwrapped by runner.chat.render."""
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(Path(model_dir))


def chat_wrapper(family: str, model_dir: Path, tokenizer=None) -> dict:
    """The Amendment 5 wrapper of one model directory: {"prompt_ids": [...],
    "record": {...}}. The prompt is runner.chat.render(family, tokenizer,
    [{"role": "user", "content": WRAP_USER_MESSAGE}], WRAP_EFFORT), i.e. the
    scored runs' rendering with the thinking-off kwargs passed explicitly.
    The record holds everything needed to rebuild it: the user message, the
    kwargs, the template's source and sha256, the rendered text's sha256,
    the prompt ids and what the prompt prefilled."""
    from runner import chat

    d = Path(model_dir)
    tok = tokenizer if tokenizer is not None else load_chat_tokenizer(d)
    messages = [{"role": "user", "content": WRAP_USER_MESSAGE}]
    text, ids, rendered_sha = chat.render(family, tok, messages, WRAP_EFFORT)
    if not ids:
        raise common.BenchError(f"{family}: the chat wrapper rendered no tokens")
    if family == "kolibri":
        tb = chat.local_template_bytes(d)
        if tb is None:
            raise common.BenchError(f"kolibri: no chat template in {common.redact_path(d)}")
        template = {"template": "the model directory's own (tokenizer_config.json chat_template)",
                    "template_sha256": common.sha256_bytes(tb)}
    else:
        par = chat.template_parity(family, d)
        template = {"template": f"runner/templates/{chat.TEMPLATE_FILES[family]}",
                    "template_sha256": chat.template_sha256(family),
                    "template_equals_model_dir": par.get("equal")}
    record = {
        "amendment": WRAP_AMENDMENT,
        "rule": WRAP_RULE,
        "messages": messages,
        "add_generation_prompt": True,
        "effort": WRAP_EFFORT,
        "kwargs": chat.template_kwargs(family, WRAP_EFFORT),
        **template,
        "rendered_sha256": rendered_sha,
        "prompt_ids": [int(i) for i in ids],
        "prompt_tokens": len(ids),
        "prompt_ids_sha256": common.sha256_ids(ids),
        "prefill": chat.prompt_prefill(family, text),
        "text_tokens": "the text tokenised alone, add_special_tokens=False (raw tokenizers, tokenizer.json)",
    }
    return {"prompt_ids": [int(i) for i in ids], "record": record, "rendered": text}


def wrapper_boundary(tokenizer_json: Path, rendered: str, text: str, prompt_ids: list[int], ids: list[int]) -> dict:
    """How the scored sequence (prompt_ids + the text tokenised alone) meets
    the raw tokenizer's joint encoding of the rendered prompt + text. Equal:
    {"joint_equals_scored": True}. Else the stretch where they differ, by
    common prefix and suffix: {"joint_equals_scored": False, "scored_ids",
    "joint_ids", "from_text_token": index of the first differing scored id
    relative to the text (-1 = the wrapper's last token)}. Amendment 5 keeps
    the scored ids (the text tokenised alone); this records the difference."""
    from tokenizers import Tokenizer

    joint = [int(i) for i in
             Tokenizer.from_file(str(tokenizer_json)).encode(rendered + text, add_special_tokens=False).ids]
    scored = [int(i) for i in prompt_ids] + [int(i) for i in ids]
    if joint == scored:
        return {"joint_equals_scored": True}
    n = min(len(joint), len(scored))
    pre = next((k for k in range(n) if joint[k] != scored[k]), n)
    suf = 0
    while suf < n - pre and joint[-1 - suf] == scored[-1 - suf]:
        suf += 1
    return {"joint_equals_scored": False, "scored_ids": scored[pre:len(scored) - suf],
            "joint_ids": joint[pre:len(joint) - suf], "from_text_token": pre - len(prompt_ids)}


def scored_input(prompt_ids: list[int], ids: list[int]) -> tuple[list[int], int]:
    """(input_ids, score_from) for teacher-forcing `ids` after `prompt_ids`:
    input = prompt_ids + ids[:-1]; logit row score_from + j predicts ids[j]
    (score_from = len(prompt_ids) - 1), so exactly len(ids) rows are scored.
    The raw-text measurement is prompt_ids = [context-start token]
    (score_from 0)."""
    if not prompt_ids:
        raise common.BenchError("teacher forcing needs at least one context token before the text")
    if not ids:
        raise common.BenchError("no text tokens to score")
    return [int(i) for i in prompt_ids] + [int(i) for i in ids[:-1]], len(prompt_ids) - 1


# ---------------------------------------------------------------------------
# Teacher forcing and KL (MLX)
# ---------------------------------------------------------------------------


def forward_chunks(model, input_ids: list[int], chunk: int = CHUNK) -> Iterator[tuple[int, object, str]]:
    """(start, logits fp32 [n, V], native logits dtype) per chunk, through a
    fresh cache of the model's own (make_prompt_cache)."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    for s in range(0, len(input_ids), chunk):
        x = mx.array(input_ids[s : s + chunk], dtype=mx.int32)[None]
        out = model(x, cache=cache)[0]
        native = str(out.dtype).replace("mlx.core.", "")
        logits = out.astype(mx.float32)
        mx.eval(logits)
        yield s, logits, native


def scored_chunks(model, input_ids: list[int], score_from: int = 0, chunk: int = CHUNK) -> Iterator[tuple[int, object, str]]:
    """forward_chunks restricted to the scored rows: (t0, logits fp32 [n, V],
    native dtype) where row r is the prediction of target t0 + r. Rows before
    score_from (the wrapper's prompt) are computed, as they must be, and
    dropped; a chunk that holds only such rows yields nothing."""
    if not 0 <= score_from < len(input_ids):
        raise common.BenchError(f"score_from {score_from} outside the {len(input_ids)} input positions")
    for s, logits, native in forward_chunks(model, input_ids, chunk):
        n = logits.shape[0]
        a = max(score_from - s, 0)
        if a >= n:
            del logits
            continue
        yield s + a - score_from, (logits[a:] if a else logits), native


def logprobs(logits_f32, round_bf16: bool):
    """log-softmax in fp32, of bf16-rounded logits (C6) or of the fp32 logits."""
    import mlx.core as mx

    x = logits_f32.astype(mx.bfloat16).astype(mx.float32) if round_bf16 else logits_f32
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def kl_rows(lp_p, lp_q):
    """KL(p || q) per row in fp32; 0 · (anything) = 0."""
    import mlx.core as mx

    p = mx.exp(lp_p)
    return mx.sum(mx.where(p > 0, p * (lp_p - lp_q), 0.0), axis=-1)


def _row_stats(lp8, lp4, targets) -> dict:
    import mlx.core as mx

    t = mx.array(np.asarray(targets, dtype=np.int32))[:, None]
    return {
        "kl": kl_rows(lp8, lp4),
        "nll8": -mx.take_along_axis(lp8, t, axis=-1)[:, 0],
        "nll4": -mx.take_along_axis(lp4, t, axis=-1)[:, 0],
        "agree": (mx.argmax(lp8, axis=-1) == mx.argmax(lp4, axis=-1)).astype(mx.float32),
    }


def _to_np(stats: dict) -> dict:
    import mlx.core as mx

    mx.eval(list(stats.values()))
    return {k: np.array(v, dtype=np.float32) for k, v in stats.items()}


def _new_acc(keys) -> dict:
    return {k: [] for k in keys}


def _finish_acc(acc: dict) -> dict:
    return {k: np.concatenate(v) if v else np.zeros(0, np.float32) for k, v in acc.items()}


def _check_targets(input_ids: list[int], targets: list[int], score_from: int) -> None:
    if len(input_ids) - score_from != len(targets):
        raise common.BenchError(f"{len(input_ids)} inputs from row {score_from} do not score {len(targets)} targets")


def compare_loaded(m8, m4, input_ids: list[int], targets: list[int], chunk: int = CHUNK, with_fp32: bool = False,
                   score_from: int = 0) -> tuple[dict, dict]:
    """Both models resident: per-position stats for one text (the rows from
    score_from on; targets[j] is the token row score_from + j predicts)."""
    _check_targets(input_ids, targets, score_from)
    keys = STATS + (("kl_fp32",) if with_fp32 else ())
    acc = _new_acc(keys)
    native = {}
    for (s, l8, n8), (s4, l4, n4) in zip(scored_chunks(m8, input_ids, score_from, chunk),
                                         scored_chunks(m4, input_ids, score_from, chunk)):
        if l8.shape != l4.shape:
            raise common.BenchError(f"logit shapes differ: {l8.shape} vs {l4.shape}")
        native = {"8bit": n8, "4bit": n4}
        n = l8.shape[0]
        for r0 in range(0, n, ROW_SLICE):
            a8, a4 = l8[r0 : r0 + ROW_SLICE], l4[r0 : r0 + ROW_SLICE]
            st = _row_stats(logprobs(a8, True), logprobs(a4, True), targets[s + r0 : s + r0 + a8.shape[0]])
            if with_fp32:
                st["kl_fp32"] = kl_rows(logprobs(a8, False), logprobs(a4, False))
            for k, v in _to_np(st).items():
                acc[k].append(v)
        del l8, l4
    return _finish_acc(acc), native


def dump_logprobs(m8, input_ids: list[int], out_dir: Path, text_id: str, chunk: int = CHUNK,
                  score_from: int = 0) -> tuple[dict, str]:
    """Write the 8-bit model's fp32 log-probs of bf16-rounded logits and of
    fp32 logits for the scored rows, [T, V] each (T = the text's tokens), as
    .npy; return {variant: path} and the native logits dtype."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"bf16": out_dir / f"K8_{text_id}.lp_bf16logits.npy", "fp32": out_dir / f"K8_{text_id}.lp_fp32logits.npy"}
    mm = {}
    native = ""
    n_rows = len(input_ids) - score_from
    for s, logits, native in scored_chunks(m8, input_ids, score_from, chunk):
        n, v = logits.shape
        if not mm:
            for variant, f in files.items():
                mm[variant] = np.lib.format.open_memmap(f, mode="w+", dtype=np.float32, shape=(n_rows, v))
        for r0 in range(0, n, ROW_SLICE):
            a = logits[r0 : r0 + ROW_SLICE]
            for variant, rnd in (("bf16", True), ("fp32", False)):
                lp = logprobs(a, rnd)
                mm[variant][s + r0 : s + r0 + a.shape[0]] = np.array(lp, dtype=np.float32)
        del logits
    for m in mm.values():
        m.flush()
    mm.clear()
    return files, native


def compare_dump(m4, input_ids: list[int], targets: list[int], files: dict, chunk: int = CHUNK,
                 score_from: int = 0) -> tuple[dict, str]:
    """4-bit model against a dump written by dump_logprobs (same input_ids
    and score_from)."""
    import mlx.core as mx

    _check_targets(input_ids, targets, score_from)
    lp8 = {variant: np.load(f, mmap_mode="r") for variant, f in files.items()}
    acc = _new_acc(STATS + ("kl_fp32",))
    native = ""
    for s, l4, native in scored_chunks(m4, input_ids, score_from, chunk):
        n, v = l4.shape
        if lp8["bf16"].shape != (len(targets), v):
            raise common.BenchError(f"dump shape {lp8['bf16'].shape} does not match ({len(targets)}, {v})")
        for r0 in range(0, n, ROW_SLICE):
            a4 = l4[r0 : r0 + ROW_SLICE]
            rows = slice(s + r0, s + r0 + a4.shape[0])
            st = _row_stats(mx.array(np.asarray(lp8["bf16"][rows])), logprobs(a4, True), targets[rows])
            st["kl_fp32"] = kl_rows(mx.array(np.asarray(lp8["fp32"][rows])), logprobs(a4, False))
            for k, v_ in _to_np(st).items():
                acc[k].append(v_)
        del l4
    return _finish_acc(acc), native


def block_sums(per_pos: dict, blocks: np.ndarray, bounds: list[int]) -> list[dict]:
    out = []
    for k in range(len(bounds) - 1):
        mask = blocks == k
        rec = {"block": k, "bytes": int(bounds[k + 1] - bounds[k]), "tokens": int(mask.sum())}
        for name, arr in per_pos.items():
            key = "agree_count" if name == "agree" else f"{name}_sum"
            rec[key] = float(arr[mask].astype(np.float64).sum())
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------


def _prepare_family(family: str, dirs: tuple[Path, Path], texts: dict, bounds: dict) -> dict:
    """Tokenise every text for one family and wrap it (Amendment 5); check
    that the 4-bit tokenizer and template give the same ids."""
    d8, d4 = dirs
    wrap = chat_wrapper(family, d8)
    if chat_wrapper(family, d4)["prompt_ids"] != wrap["prompt_ids"]:
        raise common.BenchError(f"{family}: the 8-bit and 4-bit chat wrappers render different prompt ids")
    prep = {"wrapper": {**wrap["record"], "prompt_ids_4bit_identical": True}, "texts": {},
            "tokenizer_json_sha256": common.sha256_file(d8 / "tokenizer.json")}
    same4 = True
    for t, rec in texts.items():
        ids, fb = tokenize(d8 / "tokenizer.json", rec["text"])
        if tokenize(d4 / "tokenizer.json", rec["text"])[0] != ids:
            same4 = False
        if not ids or fb[0] != 0:
            raise common.BenchError(f"{family} {t}: tokenisation does not start at byte 0")
        inputs, score_from = scored_input(wrap["prompt_ids"], ids)
        prep["texts"][t] = {
            "ids": ids,
            "inputs": inputs,
            "score_from": score_from,
            "blocks": token_blocks(fb, bounds[t]),
            "first_bytes": [int(b) for b in fb[:FIRST_ROWS + 1]],
            "boundary": wrapper_boundary(d8 / "tokenizer.json", wrap["rendered"], rec["text"], wrap["prompt_ids"], ids),
        }
    if not same4:
        raise common.BenchError(f"{family}: the 4-bit tokenizer gives different ids")
    prep["tokenizer_4bit_ids_identical"] = True
    return prep


def _committed_ids(public_dir: Path, t: str) -> Optional[list[int]]:
    files = sorted(Path(public_dir).glob(f"{t}_*.ids.json"))
    if len(files) != 1:
        return None
    obj = json.loads(files[0].read_text())
    ids = obj.get("ids") if isinstance(obj, dict) else obj
    return [int(i) for i in ids] if isinstance(ids, list) else None


def run_cell(
    results_dir: Optional[Path] = None,
    *,
    loader: Callable[[str], tuple] = genutil.load_arm,
    families=FAMILIES,
    public_dir: Optional[Path] = None,
    work_texts_dir: Optional[Path] = None,
    dump_root: Optional[Path] = None,
    chunk: int = CHUNK,
    arm_dirs: Optional[Callable[[str], Path]] = None,
) -> Path:
    """Run H8 and return the path of results/kl_8v4_<UTC>.json.

    arm_dirs(arm) -> model directory, used for tokenizers before any model is
    loaded (default common.arm_dir); the loader must load the same directory."""
    results_dir = Path(results_dir or common.default_results_dir())
    arm_dirs = arm_dirs or common.arm_dir
    common.require_identity()
    common.require_gates([a for f in families for a in common.FAMILY_PAIRS[f]])
    header = common.base_header("kl", SPEC, results_dir)
    stamp = common.utc_stamp()

    pub = Path(public_dir) if public_dir else common.EXP_DIR / "gate" / "texts"
    texts = load_texts(gate_text_paths(pub, work_texts_dir))
    verified = verify_texts(texts, manifest_text_sha256(pub))
    bounds = {t: byte_blocks(rec["text"].encode("utf-8")) for t, rec in texts.items()}
    out_texts = {
        t: {
            "file": rec["file"],
            "sha256": rec["sha256"],
            "manifest_sha256_match": verified[t],
            "bytes": rec["bytes"],
            "block_bounds": bounds[t],
            "block_bytes": [bounds[t][k + 1] - bounds[t][k] for k in range(N_BLOCKS)],
        }
        for t, rec in texts.items()
    }

    out_fam: dict[str, dict] = {}
    dump_info: dict = {}
    for family in families:
        a8, a4 = common.FAMILY_PAIRS[family]
        d8, d4 = Path(arm_dirs(a8)), Path(arm_dirs(a4))
        prep = _prepare_family(family, (d8, d4), texts, bounds)
        fam = {
            "arms": [a8, a4],
            "model_dirs": {a8: common.redact_path(d8), a4: common.redact_path(d4)},
            "fingerprints": {a8: common.model_fingerprint(a8, d8), a4: common.model_fingerprint(a4, d4)},
            "wrapper": prep["wrapper"],
            "tokenizer_json_sha256": prep["tokenizer_json_sha256"],
            "tokenizer_4bit_ids_identical": prep["tokenizer_4bit_ids_identical"],
            "texts": {},
        }
        per_pos: dict[str, dict] = {}
        native: dict[str, dict] = {}
        t0 = common.utc_iso()
        if family == "kolibri":
            # K8 alone -> dump -> unload -> K4 alone, compared with the dump.
            ddir = Path(dump_root or common.work_dir() / "kl") / stamp
            genutil.release()
            m8, _, _ = loader(a8)
            files = {}
            for t, tp in prep["texts"].items():
                files[t], n8 = dump_logprobs(m8, tp["inputs"], ddir, t, chunk, score_from=tp["score_from"])
                native[t] = {"8bit": n8}
            del m8
            genutil.release()
            dump_info = {
                "dir": common.redact_path(ddir),
                "files": {f.name: common.sha256_file(f) for fs in files.values() for f in fs.values()},
                "note": "fp32 .npy log-probs of K8, not committed",
            }
            m4, _, _ = loader(a4)
            for t, tp in prep["texts"].items():
                per_pos[t], n4 = compare_dump(m4, tp["inputs"], tp["ids"], files[t], chunk, score_from=tp["score_from"])
                native[t]["4bit"] = n4
            del m4
        else:
            genutil.release()
            m8, _, _ = loader(a8)
            m4, _, _ = loader(a4)
            for t, tp in prep["texts"].items():
                per_pos[t], native[t] = compare_loaded(m8, m4, tp["inputs"], tp["ids"], chunk,
                                                       score_from=tp["score_from"])
            del m8, m4
        genutil.release()

        tot = {"kl_sum": 0.0, "bytes": 0, "tokens": 0}
        for t, tp in prep["texts"].items():
            blocks = block_sums(per_pos[t], tp["blocks"], bounds[t])
            rec = {
                "tokens": len(tp["ids"]),
                "score_from": tp["score_from"],
                "input_ids_sha256": common.sha256_ids(tp["inputs"]),
                "target_ids_sha256": common.sha256_ids(tp["ids"]),
                "logits_dtype_native": native[t],
                "blocks": blocks,
                "wrapper_boundary": tp["boundary"],
                "first_rows": {
                    "rows": min(FIRST_ROWS, len(tp["ids"])),
                    "token_first_bytes": tp["first_bytes"],
                    **{k: [float(x) for x in v[:FIRST_ROWS]] for k, v in sorted(per_pos[t].items())},
                },
            }
            if family == "kolibri" and t in PUBLIC_TEXTS:
                committed = _committed_ids(pub, t)
                rec["ids_equal_committed_gate_ids"] = None if committed is None else committed == tp["ids"]
            fam["texts"][t] = rec
            tot["kl_sum"] += sum(b["kl_sum"] for b in blocks)
            tot["bytes"] += sum(b["bytes"] for b in blocks)
            tot["tokens"] += sum(b["tokens"] for b in blocks)
        fam["totals"] = {
            **tot,
            "kl_per_byte": tot["kl_sum"] / tot["bytes"],
            "kl_per_token": tot["kl_sum"] / tot["tokens"],
        }
        fam["t_start"], fam["t_end"] = t0, common.utc_iso()
        out_fam[family] = fam

    result = {
        **header,
        "complete": True,
        "t_end": common.utc_iso(),
        "settings": {
            "texts": list(TEXT_IDS),
            "families": list(families),
            "n_blocks": N_BLOCKS,
            "block_rule": "boundaries at whitespace near k/8 of the UTF-8 bytes (byte_blocks); a token belongs to the block holding its first byte",
            "tokenizer": "raw tokenizers, tokenizer.json of the 8-bit directory, add_special_tokens=False",
            "wrapper_rule": WRAP_RULE + " (" + WRAP_AMENDMENT + "; per family: families.<f>.wrapper)",
            "wrapper_user_message": WRAP_USER_MESSAGE,
            "wrapper_effort": WRAP_EFFORT,
            "wrapper_boundary": "families.<f>.texts.<T>.wrapper_boundary: the scored ids against the joint encoding "
                                "of rendered prompt + text; families.<f>.texts.<T>.first_rows: the stats of the first "
                                f"{FIRST_ROWS} scored rows and the first bytes of tokens 0..{FIRST_ROWS}",
            "teacher_forcing_chunk": chunk,
            "logits": "bf16-rounded then upcast to fp32 for every model (C6); kl_fp32 for Kolibri from its fp32 logits",
            "kl": "sum_v p8 * (log p8 - log p4) per position, fp32; block sums in float64",
            "normalisation": "analysis: sum KL / sum bytes (primary), sum KL / sum tokens (sensitivity)",
        },
        "texts": out_texts,
        "families": out_fam,
        "k8_dump": dump_info,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"kl_8v4_{stamp}.json"
    common.write_json_new(path, result)
    return path
