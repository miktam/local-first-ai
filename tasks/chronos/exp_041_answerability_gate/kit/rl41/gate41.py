"""
exp_041 — the answerability gates (DESIGN rev 1, SMOKE.md). Each gate maps (material, question) → P(answerable).

  G1  Qwen3-4B imatrix Q4 (exp_040's B0i GGUF) as a text model on llama-server: one chat turn, thinking off,
      max 1 token; P(yes) / (P(yes) + P(no)) from the first token's top-20 logprobs.
  G2  Kev-4B Q8_0 · G3 Clef-Flash 9B Q8_0 · G4 Clef 27B Q8_0 — native decision models on llama-server's
      /v1/systemone: one `noul` question with the material as the state.

The question text is what the voice saw: for categorical / yes-no rows it ends with the options line. The prompts
below are fixed here; a change is a re-registration. Served by llama.cpp b11711 (9d1e036), rl41/run41.sh.
"""
from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

MODELS = Path.home() / "models" / "exp041"
GATES = {
    "G1": {"kind": "text", "gguf": Path.home() / "REPOS/casasol/coapi_voice/models/qwen3-4b-b11577/qwen3-4b-Q4_K_M-imat.gguf",
           "sha": "38623d2a"},
    "G2": {"kind": "decision", "gguf": MODELS / "kev-4b/Kev-4B-Q8_0.gguf", "sha": "7c2ebed9"},
    "G3": {"kind": "decision", "gguf": MODELS / "clef-flash/Cloudflare_clef-flash-Q8_0.gguf", "sha": "55c222df"},
    "G4": {"kind": "decision", "gguf": MODELS / "clef-27b/Cloudflare_clef-Q8_0.gguf", "sha": "9fe953f0"},
}

G1_SYSTEM = "You check whether study material is enough to answer a question. Reply with one word: yes or no."
G1_USER = ("Material:\n{material}\n\nQuestion:\n{question}\n\n"
           "Does the material contain the information needed to answer this question? Answer yes or no.")
NOUL = {"type": "noul",
        "instructions": "Does the state contain the information needed to answer this question? Question: {question}",
        "criteria": {"true": "the state states the rule or figure that decides the answer",
                     "false": "the state does not settle it; answering would need facts not in the state"}}


def _post(url: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def p_yes(top_logprobs: list[dict]) -> float:
    """P(yes) / (P(yes) + P(no)) over the first token's top logprobs; 0.5 when neither word is among them."""
    py = sum(math.exp(t["logprob"]) for t in top_logprobs if t["token"].strip().lower() == "yes")
    pn = sum(math.exp(t["logprob"]) for t in top_logprobs if t["token"].strip().lower() == "no")
    return py / (py + pn) if py + pn > 0 else 0.5


def ask(gate: str, host: str, material: str, question: str) -> dict:
    """{p, tokens} — P(answerable) from one gate call."""
    if GATES[gate]["kind"] == "text":
        body = {"messages": [{"role": "system", "content": G1_SYSTEM},
                             {"role": "user", "content": G1_USER.format(material=material, question=question)}],
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": False}}
        r = _post(f"{host}/v1/chat/completions", body)
        return {"p": p_yes(r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]),
                "tokens": r.get("usage", {}).get("prompt_tokens")}
    q = {**NOUL, "instructions": NOUL["instructions"].format(question=question)}
    r = _post(f"{host}/v1/systemone", {"state": material, "questions": {"answerable": q}})
    return {"p": float(r["answers"]["answerable"]["noul"]), "tokens": r.get("usage", {}).get("input_tokens")}
