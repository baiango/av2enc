import math
W, H = 1280, 720
def luma(p):
    d=open(p,'rb').read()
    if d.startswith(b'YUV4MPEG2'): d=d[d.index(b'\n')+1:]
    return d[:W*H]
src=luma('src.i420')
def var(a,bx,by,B=8):
    v=[a[(by+y)*W+bx+x] for y in range(B) for x in range(B)]
    m=sum(v)/len(v); return sum((t-m)**2 for t in v)/len(v)
# 8x8 DCT-II basis error energy per frequency band, on textured blocks only
def basis(u,v):
    import itertools
    B=[]
    for y in range(8):
        for x in range(8):
            B.append(math.cos((2*x+1)*u*math.pi/16)*math.cos((2*y+1)*v*math.pi/16))
    return B
BAS=[[basis(u,v) for v in range(8)] for u in range(8)]
import sys
def bands(p):
    a=luma(p)
    tot={}
    for by in range(0,H,8):
        for bx in range(0,W,8):
            if var(src,bx,by)<=500: continue      # textured only
            e=[]
            for y in range(8):
                r=(by+y)*W+bx
                for x in range(8): e.append(src[r+x]-a[r+x])
            for u in range(8):
                for v in range(8):
                    c=sum(e[i]*BAS[u][v][i] for i in range(64))
                    k='DC' if (u==0 and v==0) else 'low(%d)'%(u+v) if u+v<=2 else 'mid' if u+v<=6 else 'high'
                    tot[k]=tot.get(k,0)+c*c
    s=sum(tot.values())
    return {k:100*v/s for k,v in tot.items()}
for p in sys.argv[1:]:
    b=bands(p)
    print(f'{p:14} ' + ' '.join(f'{k}={b[k]:5.1f}%' for k in ('DC','low(1)','low(2)','mid','high') if k in b))
