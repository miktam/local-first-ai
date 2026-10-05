"""Proxy on the mini (no Kolibri weights here): Qwen3-4B bf16 per-token NLL on
the gate texts T1-T4 (decoded .txt, same bytes as the peer check), under
  A  <|endoftext|> + text  (peer_check raw rule for Qwen; all text tokens scored)
  B  text alone, first token at position 0 (the gate reference's rule for
     Kolibri; tokens 1..T-1 scored)
  C  chat-wrapped (Amendment 5): assistant turn after "Write a text.", thinking off
Reports bpb, and how much of the bpb comes from the first k tokens."""
import glob, json, math, sys
import mlx.core as mx
import numpy as np
from mlx_lm import load

TEXTS = "$KIT/gate/texts"
FILES = {"T1": "T1_exp035_post.txt", "T2": "T2_malaga_ai_post.txt", "T3": "T3_de_prose_claude.txt",
         "T4": "T4_grundgesetz_art1_19.txt"}
path = glob.glob("$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/*/")[0]
model, tok = load(path)
eot = tok.convert_tokens_to_ids("<|endoftext|>")


def nll_of(prompt_ids, text_ids):
    ids = list(prompt_ids) + list(text_ids)
    x = mx.array([ids[:-1]])
    logits = model(x)[0].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    lp = np.array(lp)
    tgt = np.array(ids[1:])
    nll = -lp[np.arange(len(tgt)), tgt]
    return nll[len(prompt_ids) - 1:] if prompt_ids else nll  # rows predicting text tokens


def tbytes(ids):
    return [len(tok.decode([i]).encode("utf-8")) for i in ids]


out = {}
for t, f in FILES.items():
    text = open(f"{TEXTS}/{f}", encoding="utf-8").read()
    ids = tok.encode(text, add_special_tokens=False)
    nbytes = len(text.encode("utf-8"))
    A = nll_of([eot], ids)                       # predicts ids[0..T-1]
    B = nll_of([], ids)                          # predicts ids[1..T-1]
    msgs = [{"role": "user", "content": "Write a text."}]
    pr = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    C = nll_of(list(pr), ids)
    bA = A.sum() / math.log(2) / nbytes
    bB = B.sum() / math.log(2) / (nbytes - len(tok.decode([ids[0]]).encode()))
    bC = C.sum() / math.log(2) / nbytes
    # B aligned to A: A[1:] and B both predict ids[1..]
    first = {k: float((B[:k].sum()) / B.sum()) for k in (1, 4, 16, 64)}
    diffAB = {k: float(B[:k].sum() - A[1:k + 1].sum()) for k in (4, 16, 64, 256)}
    tail = {"A_from_tok64": float(A[64:].sum() / math.log(2) / sum(tbytes(ids[64:]))),
            "B_from_tok64": float(B[63:].sum() / math.log(2) / sum(tbytes(ids[64:])))}
    out[t] = {"tokens": len(ids), "bytes": nbytes, "bpb_A_eot": float(bA), "bpb_B_noprefix": float(bB),
              "bpb_C_chat": float(bC), "B_share_of_nll_first_k": first,
              "B_minus_A_nats_first_k": diffAB, "tail_bpb": tail,
              "A_first8": [round(float(v), 2) for v in A[:9]], "B_first8": [round(float(v), 2) for v in B[:8]]}
    print(t, json.dumps(out[t]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
