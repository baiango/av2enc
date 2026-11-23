W,H=1280,720
import math,os
def luma(p):
    d=open(p,'rb').read()
    if d.startswith(b'YUV4MPEG2'): d=d[d.index(b'\n')+1:]
    return d[:W*H]
src=luma('src.i420')
def var(bx,by,B=8):
    v=[src[(by+y)*W+bx+x] for y in range(B) for x in range(B)]
    m=sum(v)/len(v); return sum((t-m)**2 for t in v)/len(v)
# precompute texture mask (var>=500, full 8x8 blocks)
tex=[(bx,by) for by in range(0,H,8) for bx in range(0,W,8) if var(bx,by)>=500]
print(f'texture 8x8 blocks: {len(tex)}')
def tex_mse(p):
    a=luma(p); s=0
    for bx,by in tex:
        for y in range(by,by+8):
            r=y*W+bx
            for x in range(8):
                e=src[r+x]-a[r+x]; s+=e*e
    return s/(len(tex)*64)
import sys
rows=[]
for arg in sys.argv[1:]:
    name,path=arg.split('=',1)
    m=tex_mse(path); rows.append((name,os.path.getsize(path),m))
for n,b,m in rows:
    print(f'{n:12} {b:8} B  tex_psnr={10*math.log10(255*255/m):6.3f} dB  tex_mse={m:7.2f}')
