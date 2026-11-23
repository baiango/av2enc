W, H = 1280, 720
import math
def luma(p):
    d=open(p,'rb').read()
    if d.startswith(b'YUV4MPEG2'): d=d[d.index(b'\n')+1:]
    return d[:W*H]
src=luma('src.i420')
def psnr(p):
    a=luma(p); mse=sum((x-y)**2 for x,y in zip(src,a))/(W*H)
    return 10*math.log10(255*255/mse)
def var(a,bx,by,BS=8):
    v=[a[(by+y)*W+bx+x] for y in range(BS) for x in range(BS)]
    m=sum(v)/len(v); return sum((t-m)**2 for t in v)/len(v)
# per-activity PSNR
def act_psnr(p):
    a=luma(p); out={}
    for name,lo,hi in (('flat',-1,50),('mid',50,500),('texture',500,10**9)):
        s=0;c=0
        for by in range(0,H,8):
            for bx in range(0,W,8):
                v=var(src,bx,by)
                if lo<v<=hi:
                    for y in range(by,by+8):
                        r=y*W+bx
                        for x in range(8):
                            e=src[r+x]-a[r+x]; s+=e*e; c+=1
        out[name]=10*math.log10(255*255/(s/c)) if c else 0
    return out
import sys,os
for p in sys.argv[1:]:
    ap=act_psnr(p)
    print(f'{os.path.basename(p):16} all={psnr(p):6.3f}  flat={ap["flat"]:6.3f} mid={ap["mid"]:6.3f} texture={ap["texture"]:6.3f}')
