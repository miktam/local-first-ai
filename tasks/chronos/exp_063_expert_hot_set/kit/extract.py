import os, sys; sys.path.insert(0, os.path.expanduser("~/models/llama.cpp-b12676/gguf-py"))
import gguf, json
d = {}
for t in gguf.GGUFReader(sys.argv[1]).tensors:
    if "down_exps" in t.name and t.name.endswith("counts"): d[int(t.name.split(".")[1])] = [int(x) for x in t.data.ravel()]
json.dump({str(k): d[k] for k in sorted(d)}, open(sys.argv[2], "w"))
