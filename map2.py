import numpy as np
from noise import Noise

def gauss(x,y,x0=0.5,y0=0.5,sigma=0.1,amp=1):
    xf=((x-x0)**2)/(2*sigma**2)
    yf=((y-y0)**2)/(2*sigma**2)
    return amp*np.exp(-1*(xf+yf))

for seed in [1,2,3,4,5]:
    n=Noise(seed=seed)
    n.center(standardise=True)                                # center noise, make range roughly -3..3
    n.glf(lambda v,x,y: (1-gauss(x,y,amp=0.5,sigma=0.05))*v)  # bias center towards 0
    n.ptf(lambda v:v**2)                                      # square to get rid of hollows, make hills higher
    n.to_png(f"map{seed}_wm.png",f"map{seed}_hm.png")
    print(n)


