# Generates the drift carrier used by drift-carrier.txt: 60 frames 480x240
# C420jpeg, a smooth gradient for 0-29 and a high-contrast pattern for 30-59
# (a synthetic scene cut, so the reference PSNR falls below the drift anchor).
# Usage: python3 make_drift_carrier.py > /tmp/drift60.y4m
import sys
W,H,F=480,240,60
def frame(phase):
    # phase 0: smooth gradient; phase 1: horizontally inverted high-contrast pattern
    y=bytearray(W*H)
    for j in range(H):
        for i in range(W):
            v=(i*255)//(W-1) if phase==0 else (255-((i*7+j*3)%256))
            y[j*W+i]=v
    c=bytes([128])*((W//2)*(H//2))*2
    return bytes(y)+c
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/drift60.y4m"
with open(out,"wb") as f:
    f.write(b"YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420jpeg\n"%(W,H))
    for n in range(F):
        f.write(b"FRAME\n")
        f.write(frame(0 if n<30 else 1))
print("wrote " + out)
