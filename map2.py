import numpy as np
from noise import Noise


def flatten(v, size=0.1):
    if v <= -size:
        return v + size
    elif v >= size:
        return v - size
    return 0

for seed in [1,2,3,4,5]:
    print(seed)
    (Noise(seed=seed).center(standardise=True)
       .ptf(flatten).to_png(f"map{seed}_wm.png",f"map{seed}_hm.png")) 


