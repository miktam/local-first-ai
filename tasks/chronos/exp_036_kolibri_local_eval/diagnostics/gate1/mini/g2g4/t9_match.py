import sys, json
sys.path.insert(0, '$KIT')
from pathlib import Path
from gate import build_gate_text as bgt, common, textset
tok = common.load_raw_tokenizer(Path('$EXP036_MODELS/kolibri/Kolibri-1-BF16'))
pq = Path('$EXP036_MODELS/data/fineweb-2/data/deu_Latn/test/000_00000.parquet')
public, work = bgt.build_web(tok, pq)
t9 = work['T9']['ids']
d = json.load(open(sys.argv[1]))
bounds = d['bounds']
ids = {}
for t, f in textset.FILES.items():
    j = common.read_json(common.TEXTS_DIR / f)
    ids[t] = j['ids'] if 'ids' in j else j
ids['T5'] = work['T5']['ids']; ids['T6'] = work['T6']['ids']
starts = {'T1': bounds[0][0], 'T4': bounds[1][0], 'T3': bounds[2][0], 'T5': bounds[3][0], 'T6': bounds[4][0]}
for t, s in starts.items():
    a = ids[t]; b = t9[s:s + len(a)]
    n_eq = sum(x == y for x, y in zip(a, b))
    first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print(t, 'len', len(a), 'T9 offset', s, 'equal', n_eq, 'first diff at', first_diff, 'first ids', a[:3], b[:3])
