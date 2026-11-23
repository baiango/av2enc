import sys
W,H=1280,720
src=open('/tmp/q/src.i420','rb').read()[:W*H]
def var(bx,by,B=8):
    if bx+8>W or by+8>H: return -1
    v=[src[(by+y)*W+bx+x] for y in range(B) for x in range(B)]
    m=sum(v)/len(v); return sum((t-m)**2 for t in v)/len(v)
def bucket(v): return 'edge' if v<0 else 'flat' if v<50 else 'mid' if v<500 else 'texture'
for path in sys.argv[1:]:
    bits={}; tot=0
    for line in open(path):
        r,c,bw,bh,inter,skip,eob,b=line.split()
        r,c,bw,bh,b=int(r),int(c),int(bw),int(bh),float(b)
        tot+=b; x0,y0=c*4,r*4
        nbx=max(1,bw//8); nby=max(1,bh//8); share=b/(nbx*nby)
        for j in range(nby):
            for i in range(nbx):
                bits[(x0+i*8,y0+j*8)]=bits.get((x0+i*8,y0+j*8),0)+share
    agg={}
    for k,b in bits.items():
        t=bucket(var(*k)); a=agg.setdefault(t,[0.0,0]); a[0]+=b; a[1]+=1
    parts=[]
    for t in ('flat','mid','texture'):
        b,n=agg[t]; parts.append(f'{t}={b/8:7.0f}B({(b/8)/n:5.2f}/blk)')
    print(f'{path.split("/")[-1]:12} leaf_total={tot/8:8.0f}B  ' + '  '.join(parts))
