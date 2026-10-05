"""Smoke test of g5_parity_diag_mbp.py on the mini: a tiny random 'K8' arm with
the real vocabulary and tokenizer, fake T5/T6/T9 (load_real patched to skip
their MANIFEST sha check). Exercises parts A, B, C, D end to end."""
import os
import runpy
import shutil
import sys
from pathlib import Path

SP = Path(__file__).resolve().parent
KIT = Path("$KIT")
TOK = Path("$EXP036_MODELS/kolibri/Kolibri-1-BF16")
fm = SP / "fake_models"
fw = SP / "fake_work"
os.environ.update({"EXP036_MODELS": str(fm), "EXP036_WORK": str(fw), "EXP036_TOK": str(TOK),
                   "MLX_ENABLE_TF32": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
os.chdir(KIT)
sys.path[:0] = [str(KIT), str(KIT / "tests")]
import numpy as np  # noqa: E402

import tiny_checkpoint as tc  # noqa: E402

arm = fm / "Kolibri-1-MLX-8bit-g64"
if not (arm / "config.json").is_file():
    tc.write_tiny_checkpoint(arm, seed=0, preset="w513",
                             overrides={"vocab_size": 128000, "num_attention_heads": 12, "num_key_value_heads": 1,
                                        "head_dim": 128})
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(TOK / f, arm / f)
(fw / "gate_texts").mkdir(parents=True, exist_ok=True)
for t in ("T5", "T6"):
    (fw / "gate_texts" / f"{t}.txt").write_text("Dies ist ein kurzer Ersatztext fuer den Rauchtest. " * 40)

from gate import common, textset  # noqa: E402

_orig = textset.load_real


def fake_load_real(tok_dir=None, work=None, profile=None):
    rng = np.random.default_rng(5)
    ids = {t: common.read_json(textset.TEXTS_DIR / f)["ids"] for t, f in textset.FILES.items()}
    ids["T5"] = rng.integers(0, 127000, size=1536).tolist()
    ids["T6"] = rng.integers(0, 127000, size=1536).tolist()
    t9 = rng.integers(0, 127000, size=16384).tolist()
    return textset.TextSet(profile, ids, t9, [], {}, "smoke", {"smoke": True})


textset.load_real = fake_load_real
sys.argv = ["g5_parity_diag_mbp.py", "--out", str(SP / "smoke_out")] + sys.argv[1:]
runpy.run_path(str(SP / "g5_parity_diag_mbp.py"), run_name="__main__")
