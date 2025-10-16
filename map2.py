import argparse
import numpy as np
from noise import Noise


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="randon number seed, a positive int")
parser.add_argument("--it", type=int, default=3, help="number of iterations")
args = parser.parse_args()

rng=np.random.default_rng(args.seed)

for it in range(args.it):
    seed=rng.integers(low=0,high=2**16)
    map=Noise(seed=seed,
              size=4096,
              center=1024,
              octaves=9,
              base_freq=rng.uniform()+1)
    map.center(method="mean", standardise=True)
    map.center(method="center_pixel",standardise=False)
    map.ptf(lambda v: np.exp(v) if v>=0 else np.exp(3.5*v))
    map.to_png(f"map{it}_wm.png",
               f"map{it}_hm.png") # done
    print(map)



