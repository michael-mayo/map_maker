import argparse
import numpy as np
from noise import Noise


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="randon number seed, a positive int")
parser.add_argument("--it", type=int, default=3, help="number of iterations")
args = parser.parse_args()

rng=np.random.default_rng(args.seed)
def gauss(x,y,x0=0.5,y0=0.5,sigma=0.1,amp=1):
    xf=((x-x0)**2)/(2*sigma**2)
    yf=((y-y0)**2)/(2*sigma**2)
    return amp*np.exp(-1*(xf+yf))
for it in range(args.it):
    seed=rng.integers(low=0,high=2**16)
    map=Noise(seed=seed,size=4096,center=1024)
    map.center(standardise=True)
    map.ptf(lambda v: v+np.exp(v))
    x0,y0,amp,sig,k=rng.uniform(low=0,high=1,size=5)
    map.glf(lambda v,x,y:
            v+gauss(x,y,
                    x0=x0 if k>=0.5 else 0,
                    y0=y0 if k<0.5 else 0,
                    amp=amp/2.0+0.1,
                    sigma=sig/2+0.25)
            )
    map.to_png(f"map{it}_wm.png",
               f"map{it}_hm.png") # done
    print(map)


