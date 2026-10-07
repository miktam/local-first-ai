"""Reference side of the W13 spike: exp_036 reference/kolibri_ref.py (read-only,
imported from its path, no bytecode), fp32 mode, full-sequence forward from
position 0. Saves per-layer residual stream h_l (after decoder layer l), the
final-norm output and the fp32 logits."""
import json
import sys

sys.dont_write_bytecode = True
E36 = str(__import__("pathlib").Path(__file__).resolve().parents[4] / "exp_036_kolibri_local_eval")  # exp_037 copy: was the absolute path of exp_036's directory on the mini
sys.path.insert(0, E36)
import numpy as np  # noqa: E402
from reference.kolibri_ref import REF_VERSION, KolibriReference, reference_tree_sha256  # noqa: E402

V = sys.argv[1]
cks = sys.argv[2].split(",")
ids = json.load(open(f"{V}/ck/ids.json"))
print("ref", REF_VERSION, "tree", reference_tree_sha256(E36 + "/reference"))
for ck in cks:
    ref = KolibriReference(f"{V}/ck/{ck}")
    assert ref.mode == "fp32"
    for name in ("ids64", "ids600"):
        x = np.array(ids[name], dtype=np.int64)
        logits, layers, final = ref.forward(x, return_hidden=True)
        np.savez(f"{V}/out/ref_{ck}_{name}.npz", logits=logits, hidden=np.stack(layers), final=final, ids=x)
        print(ck, name, logits.shape, np.stack(layers).shape)
