"""Discrimination control (W13 spike): would this comparison have caught a
wiring misreading? The vendor's (vLLM) outputs are compared with each of the
seven exp_036 G3 reference mutants (reference/mutants.py, read-only) and with
the reference run at sliding_window W-1 and W+1 (config-only copies of the
checkpoint, same shards). Each must exceed the 1e-4 bound by a wide margin."""
import json
import sys

sys.dont_write_bytecode = True
E36 = str(__import__("pathlib").Path(__file__).resolve().parents[4] / "exp_036_kolibri_local_eval")  # exp_037 copy: was the absolute path of exp_036's directory on the mini
sys.path.insert(0, E36)
import numpy as np  # noqa: E402
from reference.kolibri_ref import KolibriReference  # noqa: E402
from reference.mutants import MUTANTS  # noqa: E402

V = sys.argv[1]
res = {}


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


for ck, seq in (("vendor_s0", "ids600"), ("rl_s29", "ids64")):
    g = np.load(f"{V}/out/vllmA_{ck}_{seq}.npz")
    ids = g["ids"]
    ref = KolibriReference(f"{V}/ck/{ck}")
    streams = ref.forward_streams(ids, MUTANTS, include_base=True)
    for name, logits in streams.items():
        if name.startswith("_"):
            continue
        res[f"{ck}:{seq}:{name}"] = rel(g["logits"], logits)
    if ck == "vendor_s0":
        for w in (64, 66):
            r2 = KolibriReference(f"{V}/ck/vendor_s0_w{w}")
            res[f"{ck}:{seq}:window{w}"] = rel(g["logits"], r2.forward(ids))
for k, v in res.items():
    print(f"{k:40s} logits rel_l2 vendor vs variant {v:.2e}")
json.dump(res, open(f"{V}/out/discrimination.json", "w"), indent=1)
