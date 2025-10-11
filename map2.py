import numpy as np
from noise import Noise



for seed in [1,2,3,4,5]:
    print(seed)
    (Noise(seed=seed)
     .center(standardise=True)
     .flatten_center()
     .ptf(lambda v: np.e**v)
     .to_png(f"map{seed}_wm.png",f"map{seed}_hm.png"))


