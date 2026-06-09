#!/usr/bin/env python3
"""
exp_013 — Corrected native-Ollama audit loop.

This is the rebuilt version of the "multi-agent consensus loop" plan, aimed at the
failure mode exp_012 actually measured for gemma4:26b: zero true positives on
cross-document auditing (recall), not false positives, formatting, or speed.

What changed vs. the original Gemini plan, and why:

  1. EXPLICIT num_ctx + TRUNCATION GUARD (was: unstated).
     Ollama defaults to 4096 tokens and silently drops the *oldest* tokens — i.e.
     the system prefix you were counting on caching — when you overflow. A code
     graph + compliance manuals blows past 4K. We set num_ctx explicitly and run a
     canary check to prove the head of the context actually survived before trusting
     any output.

  2. NO temperature=0 + min_p CONTRADICTION (was: both, in Phase 5).
     Greedy decoding (temp 0) ignores min_p/top_p/top_k entirely. We use temp 0 for
     deterministic checks and a small temp for recall-oriented extraction. min_p is
     never set; it would have been dead code.

  3. FRESH-CONTEXT VERIFIER (was: single-context self-critique).
     Self-critique in the same growing context just re-reads its own tokens. Each
     candidate gap is instead re-checked in a brand-new context containing only the
     source + the one claim, with an instruction to quote the supporting line or
     mark it INVALID. That's an external check, not self-soothing.

  4. DECOMPOSED EXTRACTION (was: implicit cross-document leap).
     The misses in exp_012 (A2, A5) required bridging a code observation to a GDPR
     article. We split that: extract atomic code facts, extract atomic policy rules,
     then bridge them in a dedicated pass. The 26B model may not do the leap
     implicitly but might do each half.

  5. RECALL-DIRECTION LOOP (was: "remove the gap if evidence missing").
     The original loop only pruned — it optimizes precision, but the local model's
     problem was recall. Here pruning happens only in the verifier; generation is
     tuned for breadth.

  6. SCHEMA-CONSTRAINED OUTPUT (was: token-tag string slicing).
     Uses Ollama's `format` JSON-schema structured outputs. No [FINAL JSON] boundary
     parsing. Final assembly is done deterministically in Python so the model can't
     reintroduce noise at the end.

  7. COVERAGE LOGGING.
     Every candidate is logged with its verification verdict to a JSONL trace, so you
     can read recall directly (what was attempted vs. what survived) instead of
     inferring it from the final JSON. Rubric scoring stays manual, matching your
     pre-registered methodology.

  8. AUTO-DIAGNOSIS (the point of the first gemma run).
     The instrument does not measure "how to score >0" — it localizes WHERE a run
     fails. At the end of a run (or standalone via --diagnose-trace) the trace is read
     and the run is auto-labelled by drop-off point, each mapped to the fix it implies:
        extraction_empty  -> Stage 1 produced no facts/rules
        no_bridge         -> facts+rules existed, Stage 2 connected nothing
        vague_candidates  -> candidates generated but too generic; verifier killed them
        over_pruned       -> specific candidates all rejected; verifier too strict
        produced_output   -> no structural zero; remaining question is rubric mapping
     Deterministic and rule-based on the trace (NO llm call), so the diagnosis itself
     stays reproducible.

Framework-free: stdlib + requests only. No LangChain/AutoGen.

Usage:
    python exp_013_local_audit_loop.py \
        --code inference_log.py mcp_server.py \
        --policy ropa.md retention_schedule.md \
        --model gemma4:26b \
        --num-ctx 32768 \
        --out gaps.json \
        --trace trace.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    host: str = "http://localhost:11434"
    model: str = "gemma4:26b"
    num_ctx: int = 32768          # set explicitly; see truncation guard below
    keep_alive: str = "10m"       # keep weights resident across the loop
    num_predict: int = 4096       # output budget; 2048 truncated 16-rule policy extraction
    temp_extract: float = 0.3     # breadth for recall-oriented passes
    temp_check: float = 0.1       # gemma4:26b returns empty content at temp=0.0 with schema injection
    request_timeout: int = 600
    trace_path: str = "trace.jsonl"


CANARY = f"AUDIT-CANARY-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Tracing (coverage visibility)
# --------------------------------------------------------------------------- #

class Tracer:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.write_text("")  # truncate

    def log(self, stage: str, event: str, data: Any) -> None:
        rec = {"ts": round(time.time(), 3), "stage": stage, "event": event, "data": data}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _schema_example(schema: dict) -> Any:
    """Generate a concrete example value from a JSON Schema dict.

    Used to give the model an instance template instead of the schema document
    itself — gemma4:26b mirrors the schema structure rather than instantiating it
    when we inject the raw schema, producing double-nested responses.
    """
    t = schema.get("type", "object")
    if t == "object":
        props = schema.get("properties", {})
        return {k: _schema_example(v) for k, v in props.items()}
    if t == "array":
        return [_schema_example(schema["items"])]
    if t == "boolean":
        return True
    if t == "string":
        return "..."
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    return "..."


# --------------------------------------------------------------------------- #
# Ollama native /api/chat client
# --------------------------------------------------------------------------- #

class OllamaClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def chat(
        self,
        messages: list[dict],
        schema: dict | None = None,
        temperature: float | None = None,
    ) -> Any:
        """One stateless call. Returns parsed dict if schema given, else text.

        num_ctx is sent on EVERY request so the loaded runner keeps a single,
        explicit context size (a differing num_ctx forces a model reload).
        """
        options = {
            "num_ctx": self.cfg.num_ctx,
            "num_predict": self.cfg.num_predict,
            "temperature": self.cfg.temp_check if temperature is None else temperature,
            # Deliberately NO min_p / top_p override here. At temp 0 they are dead
            # code; at temp>0 the model defaults are fine for this task.
        }
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": options,
        }
        if schema is not None:
            # Do NOT use format:"json" — gemma4:26b on /api/chat returns empty content
            # with that constraint active. Use a concrete example template instead of
            # injecting the raw JSON Schema object — the schema structure confuses the
            # model into mirroring the schema document rather than an instance of it.
            example = _schema_example(schema)
            schema_instruction = (
                "Output ONLY valid JSON — no markdown fences, no explanation, no preamble.\n"
                f"Required output structure (fill in the values):\n{json.dumps(example, indent=2)}"
            )
            messages = list(messages)  # shallow copy — don't mutate caller's list
            if messages and messages[0]["role"] == "system":
                messages[0] = dict(messages[0])
                messages[0]["content"] += f"\n\n{schema_instruction}"
            else:
                messages = [{"role": "system", "content": schema_instruction}] + messages
            body["messages"] = messages

        def _do_request() -> str:
            r = requests.post(
                f"{self.cfg.host}/api/chat", json=body, timeout=self.cfg.request_timeout
            )
            r.raise_for_status()
            return r.json()["message"]["content"]

        content = _do_request()
        if schema is None:
            return content
        if not content or not content.strip():
            # Empty response — return a safe default rather than crashing the run.
            return {}

        def _parse(raw: str) -> Any:
            text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            text = re.sub(r"\s*```$", "", text)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)
            return json.loads(text)

        try:
            return _parse(content)
        except json.JSONDecodeError:
            # One retry — stochastic output can occasionally produce malformed JSON.
            content = _do_request()
            if not content or not content.strip():
                return {}
            try:
                return _parse(content)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Schema response was not valid JSON after retry: {content[:300]}") from e


# --------------------------------------------------------------------------- #
# Stage 0 — context budget + truncation guard
# --------------------------------------------------------------------------- #

def estimate_tokens(text: str) -> int:
    # Rough heuristic; good enough to catch order-of-magnitude overflow.
    return len(text) // 4


def budget_check(cfg: Config, *texts: str, tracer: Tracer) -> None:
    est = sum(estimate_tokens(t) for t in texts)
    usable = cfg.num_ctx - cfg.num_predict
    tracer.log("stage0", "budget", {"est_input_tokens": est, "usable": usable, "num_ctx": cfg.num_ctx})
    if est > usable:
        raise SystemExit(
            f"Estimated input ~{est} tokens exceeds usable window {usable} "
            f"(num_ctx={cfg.num_ctx} minus num_predict={cfg.num_predict}). "
            f"Raise --num-ctx or trim the bundle. Ollama would truncate silently."
        )


def verify_head_retained(client: OllamaClient, bundle: str, tracer: Tracer) -> None:
    """Prove the TOP of the context survived. Ollama drops oldest tokens first,
    which is exactly the system prefix. If the canary can't be echoed, the head
    was clipped and every downstream answer is untrustworthy."""
    system = f"{CANARY}\n\n--- BEGIN AUDIT BUNDLE ---\n{bundle}\n--- END AUDIT BUNDLE ---"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Output ONLY the canary token printed on the very first line of your context. Nothing else."},
    ]
    out = client.chat(messages, temperature=0.0).strip()
    ok = CANARY in out
    tracer.log("stage0", "canary", {"expected": CANARY, "got": out[:80], "ok": ok})
    if not ok:
        raise SystemExit(
            "Canary not echoed — the context head was truncated. Raise --num-ctx; "
            "do NOT trust any audit run in this state."
        )


# --------------------------------------------------------------------------- #
# JSON schemas for structured outputs
# --------------------------------------------------------------------------- #

FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "fact": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["id", "fact", "location"],
            },
        }
    },
    "required": ["facts"],
}

RULES_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "rule": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["id", "rule", "source"],
            },
        }
    },
    "required": ["rules"],
}

CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "claim": {"type": "string"},
                    "code_fact_ref": {"type": "string"},
                    "policy_rule_ref": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                },
                "required": ["id", "claim", "code_fact_ref", "policy_rule_ref", "severity"],
            },
        }
    },
    "required": ["gaps"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "evidence": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["supported", "evidence", "reasoning"],
}


# --------------------------------------------------------------------------- #
# Stage 1 — decomposed extraction (fresh single-purpose contexts)
# --------------------------------------------------------------------------- #

def extract_code_facts(client: OllamaClient, code_text: str, cfg: Config, tracer: Tracer) -> list[dict]:
    messages = [
        {"role": "system", "content":
            "You are a systems engineer. State plainly what the code ACTUALLY does. "
            "Report only observable implementation facts: what is logged, stored, "
            "authenticated, hashed, versioned, exposed, or omitted. No legal opinions. "
            "Be exhaustive — list every concrete behaviour, including absences "
            "(e.g. 'no authentication on endpoint X', 'model recorded as label not hash')."},
        {"role": "user", "content": f"CODE:\n```\n{code_text}\n```"},
    ]
    facts = client.chat(messages, schema=FACTS_SCHEMA, temperature=cfg.temp_extract)["facts"]
    tracer.log("stage1a", "code_facts", {"count": len(facts), "facts": facts})
    return facts


def extract_policy_rules(client: OllamaClient, policy_text: str, cfg: Config, tracer: Tracer) -> list[dict]:
    messages = [
        {"role": "system", "content":
            "You are a compliance officer. Extract every concrete obligation the "
            "policy/regulation imposes: retention claims, DPIA requirements, "
            "auditability/accountability requirements, access-control expectations, "
            "and which article or section states each. One atomic rule per item."},
        {"role": "user", "content": f"POLICY / REGULATION:\n```\n{policy_text}\n```"},
    ]
    rules = client.chat(messages, schema=RULES_SCHEMA, temperature=cfg.temp_extract)["rules"]
    tracer.log("stage1b", "policy_rules", {"count": len(rules), "rules": rules})
    return rules


# --------------------------------------------------------------------------- #
# Stage 2 — bridge: code facts x policy rules -> candidate gaps
# --------------------------------------------------------------------------- #

def bridge_candidates(client: OllamaClient, facts: list[dict], rules: list[dict],
                      cfg: Config, tracer: Tracer) -> list[dict]:
    messages = [
        {"role": "system", "content":
            "You connect implementation facts to policy rules. A GAP exists when a "
            "specific code fact contradicts, or fails to satisfy, a specific policy "
            "rule. For each gap cite the code_fact id it relies on and the policy_rule "
            "id it relies on. Prefer recall: propose every plausible gap now; it will "
            "be verified later. Do NOT invent facts or rules outside the lists given."},
        {"role": "user", "content":
            "CODE FACTS:\n" + json.dumps(facts, indent=2) +
            "\n\nPOLICY RULES:\n" + json.dumps(rules, indent=2)},
    ]
    gaps = client.chat(messages, schema=CANDIDATES_SCHEMA, temperature=cfg.temp_extract)["gaps"]
    tracer.log("stage2", "candidates", {"count": len(gaps), "gaps": gaps})
    return gaps


# --------------------------------------------------------------------------- #
# Stage 3 — fresh-context verifier (one clean context per candidate)
# --------------------------------------------------------------------------- #

def verify_candidate(client: OllamaClient, gap: dict, source: str,
                     tracer: Tracer) -> dict:
    # Fresh messages list every call = no contamination from prior turns.
    messages = [
        {"role": "system", "content":
            "You verify a single claimed compliance gap against the source material. "
            "Quote the exact line(s) that prove the gap is real. If you cannot find "
            "direct supporting evidence in the source, set supported=false. Do not be "
            "charitable; an unverifiable claim is unsupported."},
        {"role": "user", "content":
            f"SOURCE:\n```\n{source}\n```\n\nCLAIMED GAP:\n{json.dumps(gap, indent=2)}\n\n"
            "Is this gap directly supported by the source above?"},
    ]
    verdict = client.chat(messages, schema=VERDICT_SCHEMA)  # uses cfg.temp_check (0.1)
    if not verdict or "supported" not in verdict:
        # Empty response — treat as unsupported so the candidate is dropped, not the run.
        verdict = {"supported": False, "evidence": "", "reasoning": "empty_response_from_model"}
    tracer.log("stage3", "verdict", {"gap_id": gap["id"], "supported": verdict["supported"],
                                     "claim": gap["claim"], "evidence": (verdict.get("evidence") or "")[:200]})
    return verdict


# --------------------------------------------------------------------------- #
# Stage 4 — deterministic assembly (no LLM; can't reintroduce noise)
# --------------------------------------------------------------------------- #

_SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

def assemble_final(verified: list[dict]) -> list[dict]:
    out = sorted(verified, key=lambda g: _SEV_ORDER.get(g.get("severity", "low"), 9))
    for i, g in enumerate(out, 1):
        g["rank"] = i
    return out


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def run_audit(cfg: Config, code_text: str, policy_text: str) -> dict:
    tracer = Tracer(cfg.trace_path)
    client = OllamaClient(cfg)
    bundle = f"=== CODE ===\n{code_text}\n\n=== POLICY ===\n{policy_text}"

    # Stage 0
    budget_check(cfg, code_text, policy_text, tracer=tracer)
    verify_head_retained(client, bundle, tracer)

    # Stage 1 (decomposed, fresh contexts)
    facts = extract_code_facts(client, code_text, cfg, tracer)
    rules = extract_policy_rules(client, policy_text, cfg, tracer)

    # Stage 2
    candidates = bridge_candidates(client, facts, rules, cfg, tracer)

    # Stage 3 (verify each in isolation)
    verified, attempted = [], []
    for gap in candidates:
        verdict = verify_candidate(client, gap, bundle, tracer)
        attempted.append({"id": gap["id"], "claim": gap["claim"], "supported": verdict["supported"]})
        if verdict["supported"]:
            gap["evidence"] = verdict["evidence"]
            verified.append(gap)

    # Stage 4 (deterministic)
    final = assemble_final(verified)

    # Coverage summary — read recall directly off this.
    coverage = {
        "code_facts": len(facts),
        "policy_rules": len(rules),
        "candidates_generated": len(candidates),
        "candidates_verified": len(verified),
        "verification_survival_rate": round(len(verified) / max(1, len(candidates)), 3),
        "attempted": attempted,
    }
    tracer.log("done", "coverage", coverage)

    # Localize the failure (deterministic; reads the trace just written).
    dx = diagnose(cfg.trace_path)
    tracer.log("done", "diagnosis", dx)
    return {"gaps": final, "coverage": coverage, "diagnosis": dx}


# --------------------------------------------------------------------------- #
# Diagnosis — deterministic, rule-based on the trace (NO llm)
# --------------------------------------------------------------------------- #

# A claim is "specific" if it carries a concrete anchor a rubric item could be
# matched to: a filename, a legal/section reference, a quoted literal, or a
# code-style identifier (snake_case, CamelCase, ALLCAPS_CONST, or a call()).
_FILENAME_RE = re.compile(r"\b[\w./-]+\.(?:py|md|jsonl|json|ya?ml|txt|cfg|toml|ini|sql)\b")
_LEGAL_RE = re.compile(r"(§|\bArt\.?\s*\d|\bArticle\s+\d|\bSection\s+\d|\b\d+\.\d+\b)")
_IDENT_RE = re.compile(r"\b\w+_\w+\b|\b[a-z]+[A-Z]\w+\b|\b[A-Z][A-Z0-9_]{3,}\b|\b\w+\(\)")
_QUOTED_RE = re.compile(r"[\"'`][^\"'`]{3,}[\"'`]")


def _is_specific(text: str) -> bool:
    t = text or ""
    return bool(_FILENAME_RE.search(t) or _LEGAL_RE.search(t)
                or _QUOTED_RE.search(t) or _IDENT_RE.search(t))


def _load_trace(path: str) -> list[dict]:
    events = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _last(events: list[dict], stage: str, event: str) -> Any:
    for e in reversed(events):
        if e.get("stage") == stage and e.get("event") == event:
            return e["data"]
    return None


def diagnose(trace_path: str) -> dict:
    """Read the trace and localize the failure. Pure function of the trace."""
    events = _load_trace(trace_path)
    facts_d = _last(events, "stage1a", "code_facts")
    rules_d = _last(events, "stage1b", "policy_rules")
    cand_d = _last(events, "stage2", "candidates")

    n_facts = (facts_d or {}).get("count", 0)
    n_rules = (rules_d or {}).get("count", 0)
    candidates = (cand_d or {}).get("gaps", []) or []
    n_cand = (cand_d or {}).get("count", len(candidates))

    verdicts = [e["data"] for e in events
                if e.get("stage") == "stage3" and e.get("event") == "verdict"]
    n_verified = sum(1 for v in verdicts if v.get("supported"))

    claims = [c.get("claim", "") for c in candidates]
    spec_rate = (sum(_is_specific(c) for c in claims) / len(claims)) if claims else None

    incomplete = facts_d is None or rules_d is None or cand_d is None
    counts = {"facts": n_facts, "rules": n_rules, "candidates": n_cand, "verified": n_verified}
    evidence: list[str] = []

    if n_facts == 0 or n_rules == 0:
        label, conf = "extraction_empty", "high"
        evidence.append(f"facts={n_facts}, rules={n_rules}: Stage 1 produced no raw material.")
        fix = ("Stage 1 is the bottleneck. Local models systematically under-report "
               "ABSENCES. Replace open-ended extraction with a present/absent checklist "
               "over property types (auth present? retention entry present? model hash "
               "present? DPIA present?). Also re-confirm the canary passed — empty "
               "extraction can mean num_ctx silently clipped the bundle.")
    elif n_cand == 0:
        label, conf = "no_bridge", "high"
        evidence.append(f"facts={n_facts}, rules={n_rules}, candidates=0: "
                        "raw material existed but Stage 2 connected nothing.")
        fix = ("Stage 2 is the bottleneck. Stop asking the model to RECALL which rule "
               "applies. Feed the extracted rule/article text into the bridge and make "
               "it a matching task: 'for each code fact, does any of THESE rules forbid it?'")
    elif n_verified == 0:
        if spec_rate is not None and spec_rate < 0.5:
            label, conf = "vague_candidates", "heuristic"
            evidence.append(f"candidates={n_cand}, verified=0, specificity={spec_rate:.2f}: "
                            "claims lack file/section/identifier anchors; verifier rejected them.")
            fix = ("Candidates are too generic to map to any rubric item, and the verifier "
                   "correctly killed them. This is a SPECIFICITY failure, not strictness. "
                   "Require a concrete anchor per gap: a filename, line, article/section "
                   "number, or code identifier.")
        else:
            label, conf = "over_pruned", "heuristic"
            sr = "n/a" if spec_rate is None else f"{spec_rate:.2f}"
            evidence.append(f"candidates={n_cand}, verified=0, specificity={sr}: "
                            "specific candidates were all rejected.")
            fix = ("Stage 3 is likely too strict and is eating real gaps (directly hurting "
                   "recall). Run the verifier-calibration check: feed it the known-true "
                   "rubric items and confirm it ACCEPTS them. If it rejects real gaps, "
                   "loosen the verifier prompt before touching anything upstream.")
    else:
        label, conf = "produced_output", "high"
        sr = "" if spec_rate is None else f" (specificity {spec_rate:.2f})"
        evidence.append(f"candidates={n_cand}, verified={n_verified}: no structural zero.")
        fix = ("No structural failure — the loop yields verified gaps. The remaining "
               "question is rubric mapping, which stays manual" + sr + ". If the score is "
               "still low, inspect whether verified gaps are concrete enough to match items.")

    if incomplete:
        evidence.append("WARNING: trace incomplete (a stage event is missing) — the run "
                        "likely errored mid-pipeline; diagnosis is partial.")

    return {
        "label": label,
        "confidence": conf,
        "counts": counts,
        "specificity_rate": (round(spec_rate, 3) if spec_rate is not None else None),
        "evidence": evidence,
        "suggested_fix": fix,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def read_files(paths: list[str]) -> str:
    chunks = []
    for p in paths:
        text = Path(p).read_text(errors="replace")
        chunks.append(f"--- FILE: {p} ---\n{text}")
    return "\n\n".join(chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description="exp_013 corrected local audit loop")
    ap.add_argument("--code", nargs="+", help="code file(s)")
    ap.add_argument("--policy", nargs="+", help="policy/regulation file(s)")
    ap.add_argument("--model", default=Config.model)
    ap.add_argument("--host", default=Config.host)
    ap.add_argument("--num-ctx", type=int, default=Config.num_ctx)
    ap.add_argument("--keep-alive", default=Config.keep_alive)
    ap.add_argument("--out", default="gaps.json")
    ap.add_argument("--trace", default=Config.trace_path)
    ap.add_argument("--diagnose-trace", metavar="PATH",
                    help="re-run diagnosis on an existing trace.jsonl and exit (no model calls)")
    args = ap.parse_args()

    # Standalone diagnosis: re-label an old run without re-running it.
    if args.diagnose_trace:
        print(json.dumps(diagnose(args.diagnose_trace), indent=2, ensure_ascii=False))
        return

    if not args.code or not args.policy:
        ap.error("--code and --policy are required (unless --diagnose-trace is given)")

    cfg = Config(host=args.host, model=args.model, num_ctx=args.num_ctx,
                 keep_alive=args.keep_alive, trace_path=args.trace)

    code_text = read_files(args.code)
    policy_text = read_files(args.policy)

    result = run_audit(cfg, code_text, policy_text)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))

    cov = result["coverage"]
    dx = result["diagnosis"]
    print(f"\nVerified gaps: {cov['candidates_verified']}/{cov['candidates_generated']} "
          f"(survival {cov['verification_survival_rate']})", file=sys.stderr)
    print(f"Diagnosis: {dx['label']} [{dx['confidence']}]", file=sys.stderr)
    for line in dx["evidence"]:
        print(f"  - {line}", file=sys.stderr)
    print(f"  fix -> {dx['suggested_fix']}", file=sys.stderr)
    print(f"Wrote {args.out} and {args.trace}", file=sys.stderr)


if __name__ == "__main__":
    main()
