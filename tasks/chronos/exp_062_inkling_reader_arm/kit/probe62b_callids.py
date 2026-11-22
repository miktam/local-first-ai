#!/usr/bin/env python3
"""exp_062 Addendum B: what the looped models' chat template does with a repeated tool-call id (ADDENDUM_B.md, the
second stop). Made-up conversation, no generation. Output: dev_b/callid-probe-20261122.txt."""
import json, sys, urllib.request
H = "http://127.0.0.1:18461"   # a server started by hand: llama-server -m <the research model> -c 8192 --special --port 18461
def post(p, b):
    r = urllib.request.Request(H + p, json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())
TOOLS = [{"type": "function", "function": {"name": n, "description": "x", "parameters": {"type": "object", "properties": {"address": {"type": "string"}}, "required": ["address"]}}} for n in ("lookup_contact", "domain_info", "search_log")]
def convo(ids):
    m = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "event one"}]
    for i, name in zip(ids, ("lookup_contact", "domain_info", "search_log", "lookup_contact", "domain_info")):
        m.append({"role": "assistant", "content": "", "tool_calls": [{"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps({"address": "a@made-up.example"})}}]})
        m.append({"role": "tool", "tool_call_id": i, "content": json.dumps({"found": False})})
    return m
out = {}
for label, ids in (("same", ["call_1"] * 5), ("unique", [f"call_{n}" for n in range(1, 6)])):
    for n in range(1, 6):
        p = post("/apply-template", {"messages": convo(ids)[: 2 + 2 * n], "tools": TOOLS})["prompt"]
        out[(label, n)] = (len(post("/tokenize", {"content": p, "parse_special": True})["tokens"]), p)
for n in range(1, 6):
    print(n, "calls: tokens with the same id", out[("same", n)][0], "| unique ids", out[("unique", n)][0], "| difference", out[("same", n)][0] - out[("unique", n)][0])
open("same5.txt", "w").write(out[("same", 5)][1]); open("unique5.txt", "w").write(out[("unique", 5)][1])
