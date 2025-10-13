import numpy as np
from noise import Noise

def gauss(x,y,x0=0.5,y0=0.5,sigma=0.1,amp=1):
    xf=((x-x0)**2)/(2*sigma**2)
    yf=((y-y0)**2)/(2*sigma**2)
    return amp*np.exp(-1*(xf+yf))

rng=np.random.default_rng(42)

for it in range(5):
    seed0,seed1=rng.integers(low=0,high=2**16,size=2)
    land=Noise(seed=seed0).center()                       # create noise map centered on 0
    land.ptf(lambda v: 1.0 if v>=0 else 0.0).smooth()     # binarize, approx 50% 1s 50% 0s, then smooth
    map=Noise(seed=seed1)                                 # create another noise map
    map.center(standardise=True)                          # adjust second map's range to roughly -3..3
    map.glf(lambda v,x,y: (1-gauss(x,y,amp=0.5,sigma=0.05))*v)# flatten center of second map
    map.ptf(lambda v:v**2)                                # square gets rid of hollows, makes mountains higher
    map.max(land.mlt_k(0.1))                              # max aggregate the two maps
    map.to_png(f"map{it}_wm.png",
               f"map{it}_hm.png") # done
    print(map)


