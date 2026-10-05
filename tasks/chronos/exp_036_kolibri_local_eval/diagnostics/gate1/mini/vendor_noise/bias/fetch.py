import json, struct, subprocess, numpy as np
R='https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16/resolve/main/'
def rng(f,a,b):
    return subprocess.run(['curl','-sfL','-r',f'{a}-{b}',R+f],capture_output=True,check=True).stdout
shards=json.load(open('bias_shards.json'))
hdrs={}
out={}
for l in range(50):
    f=shards[str(l)]
    if f not in hdrs:
        n=struct.unpack('<Q',rng(f,0,7))[0]
        hdrs[f]=(n,json.loads(rng(f,8,8+n-1)))
    n,h=hdrs[f]
    res={}
    for key in (f'model.layers.{l}.moe.router.expert_bias',):
        a,b=h[key]['data_offsets']
        raw=rng(f,8+n+a,8+n+b-1)
        u=np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)<<16
        res['bias']=u.view(np.float32).tolist()
    out[l]=res
json.dump(out,open('bias_values.json','w'))
print('ok',len(out))
