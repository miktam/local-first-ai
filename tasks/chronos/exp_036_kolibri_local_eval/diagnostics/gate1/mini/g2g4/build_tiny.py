import sys, json, io, contextlib, inspect
from pathlib import Path
KIT = Path('$KIT')
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(KIT / 'tests'))
import tiny_checkpoint as tc
from port import convert
from tokenizers import Tokenizer, models, pre_tokenizers
root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
src = tc.write_tiny_checkpoint(root / 'Kolibri-1-BF16', seed=0, preset=sys.argv[2] if len(sys.argv) > 2 else 'pattern5', copy_port=False)
cfg = json.loads((src / 'config.json').read_text()); cfg.pop('model_file', None)
(src / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
tok = Tokenizer(models.WordLevel(vocab={f't{i}': i for i in range(tc.VOCAB_SIZE)}, unk_token='t0'))
tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit(); tok.save(str(src / 'tokenizer.json'))
(src / 'tokenizer_config.json').write_text(json.dumps({'tokenizer_class': 'PreTrainedTokenizerFast', 'eos_token': f't{tc.EOS_TOKEN_ID}', 'pad_token': f't{tc.PAD_TOKEN_ID}'}))
kw = {'guard': False} if 'guard' in inspect.signature(convert.convert).parameters else {}
for bits in (8, 4):
    with contextlib.redirect_stdout(io.StringIO()):
        convert.convert(src, root / f'Kolibri-1-MLX-{bits}bit-g64', bits, 64, **kw)
print('built', root)
