#!/usr/bin/env python3
"""exp_052: freeze what the live bot would send its answering model, once, for every question.

Runs the bot's own code (scripts/telegram_bot.py) on each question as a first message in a fresh session in the
question's language:
- the branches before routing: social closer, witness contribution, the COAPI keyword;
- the router (gemma4:e4b via Ollama, as live);
- the broad and holiday checks;
- retrieval with its empty-property cascade and double-miss stop.

Where the bot would call gemma4:26b, `ollama_chat` is replaced by a recorder, so `generate_response` builds the
exact messages and the recorder keeps them. Questions the bot answers without the model are kept with the reason.
The output holds retrieved text from the bot's collections: it stays on the mini (private).

    cd ~/REPOS/casasol && HF_HUB_OFFLINE=1 .venv/bin/python3 <exp052>/kit/freeze52.py <questions.jsonl> <frozen.jsonl>
"""
import json
import sys
from pathlib import Path

CASASOL = Path.home() / "REPOS/casasol"
sys.path.insert(0, str(CASASOL))
from scripts import telegram_bot as tb  # noqa: E402

SOCIAL_CLOSERS = {"will do", "ok", "okay", "thanks", "thank you", "great", "perfect", "got it", "sure", "sounds good",
                  "noted", "alright", "all right", "cool", "good", "nice", "wonderful", "excellent", "understood",
                  "👍", "🙏", "👌"}          # copied from handle_message


def freeze_one(q: dict) -> dict:
    text, lang = q["text"], q["lang"]
    rec = {"id": q["id"], "lang": lang, "intent_written": q["intent"], "text": text}
    stripped = text.strip().lower()
    if len(stripped.split()) <= 5 and set(stripped.split()) & SOCIAL_CLOSERS:
        return {**rec, "static": "social_closer"}
    if tb.is_witness_contribution_query(text):
        return {**rec, "static": "witness_invite"}
    if "coapi" in text.lower():
        intent, phrasing, query = "process", "coapi professional training", text
        chunks, collection, hit = tb.retrieve("process", text, lang)
        empty = False
    else:
        intent, phrasing = tb.route_intent(text)
        query = text
        if intent == "out_of_scope":
            return {**rec, "intent_routed": intent, "static": "out_of_scope"}
        if intent == "property_search" and tb.is_broad_property_query(query):
            return {**rec, "intent_routed": intent, "static": "broad_property_overview"}
        if intent == "property_search" and tb.is_holiday_rental_query(query):
            return {**rec, "intent_routed": intent, "static": "holiday_redirect"}
        chunks, collection, hit = tb.retrieve(intent, query, lang)
        empty = False
        if intent == "property_search" and not chunks:
            empty = True
            chunks, collection, hit = tb.retrieve("area_knowledge", query, lang)
        if intent == "property_search" and empty and not chunks:
            return {**rec, "intent_routed": intent, "static": "no_listings"}
    captured = {}

    def recorder(model, messages, temperature=0.2):
        captured.update(model=model, messages=messages, temperature=temperature)
        return ""

    real = tb.ollama_chat
    tb.ollama_chat = recorder
    try:
        tb.generate_response(text, {"language": lang, "last_5_turns": []}, chunks, intent, empty, "exp052")
    finally:
        tb.ollama_chat = real
    prefix = tb.EMPTY_PROPERTIES_FALLBACK.get(lang, "") if empty else ""
    return {**rec, "intent_routed": intent, "phrasing": phrasing, "collection": collection, "n_chunks": len(chunks),
            "empty_properties": empty, "static": None, "model": captured["model"],
            "temperature": captured["temperature"], "messages": captured["messages"], "prefix": prefix}


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    out = [freeze_one(json.loads(line)) for line in open(src) if line.strip()]
    with open(dst, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    reached = sum(r["static"] is None for r in out)
    print(f"{len(out)} questions, {reached} reach the answering model; "
          f"static: {sorted({r['static'] for r in out if r['static']})}")


if __name__ == "__main__":
    main()
