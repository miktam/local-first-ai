"""Decoded gate texts T1-T9 (strings) for re-tokenising with a stand-in model. Scratch only."""
import sys, json
K = '$KIT'
sys.path.insert(0, K)
from pathlib import Path
from gate import build_gate_text as bgt, common
tok = common.load_raw_tokenizer(Path('$EXP036_MODELS/kolibri/Kolibri-1-BF16'))
pq = Path('$EXP036_MODELS/data/fineweb-2/data/deu_Latn/test/000_00000.parquet')
public, work = bgt.build_web(tok, pq)
m = json.load(open(K + '/gate/texts/MANIFEST.json'))
assert public['T9']['ids_sha256'] == m['web']['T9']['ids_sha256']
out = {}
names = {'T1': 'T1_exp035_post', 'T2': 'T2_malaga_ai_post', 'T3': 'T3_de_prose_claude', 'T4': 'T4_grundgesetz_art1_19'}
for t, n in names.items():
    ids = json.load(open(f'{K}/gate/texts/{n}.ids.json'))
    ids = ids['ids'] if isinstance(ids, dict) else ids
    out[t] = tok.decode(ids)
for t in ('T5', 'T6'):
    out[t] = tok.decode(work[t]['ids'])
for t, f in (('T7', 'T7_chat_en_high.json'), ('T8', 'T8_chat_de_none.json')):
    d = json.load(open(f'{K}/gate/texts/{f}'))
    out[t] = tok.decode(d['ids'])
out['T9'] = tok.decode(work['T9']['ids'])
json.dump(out, open('texts.json', 'w'))
print({k: len(v) for k, v in out.items()})
