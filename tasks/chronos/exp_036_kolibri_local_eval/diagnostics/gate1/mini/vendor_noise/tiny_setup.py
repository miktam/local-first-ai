import sys, io, contextlib
from pathlib import Path
K = Path.cwd(); sys.path.insert(0, str(K)); sys.path.insert(0, str(K / "tests"))
from tools.precision import ensure_exact_fp32; ensure_exact_fp32()
import test_gate_drivers_tiny as t
root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
t.build_tiny_models(root)
from gate import common, run_gate
from gate.checks import ref_pass
ctx = run_gate.GateContext(models_dir=root, work_dir=root / "work", thresholds=common.load_thresholds(), tiny=True,
                           results_dir=root / "results", head_policy="quantised_head", subprocess_phases=False)
ts = run_gate.textset_for(ctx)
rec = ref_pass.fp32_dump(ctx.bf16_dir, ts.packed8() + [ts.t9], run_gate.dump_dir(ctx, ts), ts.key, log=lambda m: None)
print("dump ok", rec.get("num_tokens"), ts.profile)
