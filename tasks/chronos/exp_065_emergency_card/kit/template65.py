"""exp_065, dev: take the chat template that llama-server reports for the QAT gemma file (which chats cleanly) and try
it, with --chat-template-file, on the three gemma files from Ollama's library that did not.
    python3.14 -I kit/template65.py kit/templates/gemma4-from-qat.jinja > dev/template-<date>.txt
Two made-up prompts per file. Prints finish reason, the reply's first characters and the length of the separated reasoning."""
import json, subprocess, sys, time, urllib.request
from pathlib import Path
HOME=Path.home(); SERVER=HOME/"models/llama.cpp-b12676/build/bin/llama-server"; PORT=18466
B=HOME/".ollama/models/blobs"
def blob(tag):
    out=subprocess.run(["ollama","show",tag,"--modelfile"],capture_output=True,text=True).stdout
    fs=[Path(l.split(None,1)[1].strip()) for l in out.splitlines() if l.startswith("FROM /")]
    return max(fs,key=lambda p:p.stat().st_size)
def serve(path, extra=()):
    log=open("/tmp/tmpl65.log","wb")
    p=subprocess.Popen([str(SERVER),"-m",str(path),"-c","8192","-np","1","-ngl","99","--host","127.0.0.1","--port",str(PORT),"--jinja",*extra],stdout=log,stderr=subprocess.STDOUT)
    t0=time.time()
    while time.time()-t0<240 and p.poll() is None:
        try:
            if b'"ok"' in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",timeout=3).read(): return p
        except Exception: time.sleep(2)
    p.terminate(); return None
def stop(p):
    p.terminate()
    try: p.wait(timeout=60)
    except Exception: p.kill()
    time.sleep(2)
def chat(q, mt=400):
    req=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",json.dumps({"messages":[{"role":"user","content":q}],"temperature":0,"seed":1,"max_tokens":mt}).encode(),{"Content-Type":"application/json"})
    m=json.loads(urllib.request.urlopen(req,timeout=300).read())["choices"][0]
    return m["finish_reason"], (m["message"].get("content") or "")[:90], len(m["message"].get("reasoning_content") or "")
out=Path(sys.argv[1])
p=serve(blob("gemma4:26b-a4b-it-qat"))
props=json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props",timeout=10).read())
tpl=props.get("chat_template") or ""
out.write_text(tpl); print("QAT template chars",len(tpl),"| QAT:",chat("Reply with the single word: ready"), chat("What is 17 times 23? Answer with the number only."))
stop(p)
for tag in ("gemma4:26b","gemma4:31b","gemma4:e4b"):
    p=serve(blob(tag),("--chat-template-file",str(out)))
    if not p: print(tag,"did not start with the borrowed template"); continue
    print(tag,"| borrowed template:",chat("Reply with the single word: ready"), chat("What is 17 times 23? Answer with the number only."))
    stop(p)
