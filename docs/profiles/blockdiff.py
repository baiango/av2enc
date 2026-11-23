W, H = 1280, 720
def luma(path):
    d=open(path,'rb').read()
    if d.startswith(b'YUV4MPEG2'): d=d[d.index(b'\n')+1:]
    return d[:W*H]
src=luma('src.i420'); ours=luma('ours.y4m'); avm=luma('avm_raw.yuv')
BS=8
def err(a,b):
    out={}
    for by in range(0,H,BS):
        base_o=by*W
        for bx in range(0,W,BS):
            s=0
            for y in range(by,by+BS):
                r=y*W+bx
                for x in range(BS):
                    e=a[r+x]-b[r+x]; s+=e*e
            out[(bx,by)]=s/(BS*BS)
    return out
eo=err(src,ours); ea=err(src,avm); n=len(eo)
mo=sum(eo.values())/n; ma=sum(ea.values())/n
print(f'mean block MSE: ours={mo:.2f} ({__import__("math").log10(255*255/mo)*10:.2f} dB) '
      f'AVM={ma:.2f} ({__import__("math").log10(255*255/ma)*10:.2f} dB) ratio={mo/ma:.2f}x')
def var(k):
    bx,by=k; v=[src[(by+y)*W+bx+x] for y in range(BS) for x in range(BS)]
    m=sum(v)/len(v); return sum((t-m)**2 for t in v)/len(v)
buck={}
for k in eo:
    v=var(k); b='flat<50' if v<50 else 'mid<500' if v<500 else 'texture'
    buck.setdefault(b,[0.0,0.0,0])
    buck[b][0]+=eo[k]; buck[b][1]+=ea[k]; buck[b][2]+=1
print('\nby source activity:')
for b,(o,a,c) in sorted(buck.items()):
    print(f'  {b:10} n={c:5} ours={o/c:8.2f} AVM={a/c:8.2f} ratio={o/c/(a/c):5.2f}x')
worse=sorted(((k,eo[k],ea[k]) for k in eo if eo[k]>max(3*ea[k],20)),key=lambda t:-(t[1]-t[2]))
excess=sum(w[1]-w[2] for w in worse)
print(f'\nblocks >3x and >20 MSE worse: {len(worse)} ({100*len(worse)/n:.1f}%), '
      f'carry {100*excess/(n*(mo-ma)):.1f}% of the total excess MSE')
print('worst 8:')
for k,o,a in worse[:8]:
    print(f'  ({k[0]:4},{k[1]:3}) ours={o:9.1f} avm={a:8.1f} var={var(k):8.0f}')
