import sys, json, hashlib
sys.path.insert(0, '$KIT')
from pathlib import Path
from gate import build_gate_text as bgt, common
tok = common.load_raw_tokenizer(Path('$EXP036_MODELS/kolibri/Kolibri-1-BF16'))
pq = Path('$EXP036_MODELS/data/fineweb-2/data/deu_Latn/test/000_00000.parquet')
public, work = bgt.build_web(tok, pq)
m = json.load(open('$KIT/gate/texts/MANIFEST.json'))
print('T9 sha match', public['T9']['ids_sha256'] == m['web']['T9']['ids_sha256'])
ids = work['T9']['ids']
sep = bgt.encode(tok, bgt.SEPARATOR)
print('sep ids', sep)
pos = 0; bounds = []
for i, c in enumerate(public['T9']['components']):
    if i: pos += len(sep)
    bounds.append((pos, pos + c['n_tokens'], c['what'], c.get('row')))
    pos += c['n_tokens']
json.dump({'ids': ids, 'bounds': bounds}, open(sys.argv[1], 'w'))
for b in bounds: print(b)
