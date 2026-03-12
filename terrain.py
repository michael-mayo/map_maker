"""

pip3 install git+https://github.com/pvigier/perlin-numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)
from scipy.ndimage import gaussian_filter
import cv2
from typing import List,Tuple

def create_terrain()->np.ndarray:
    large = generate_fractal_noise_2d(
        (512, 512),
        (2, 2),
        octaves=2,
        persistence=0.5,
        lacunarity=2
    )

    small = generate_fractal_noise_2d(
        (512, 512),
        (8, 8),
        octaves=3,
        persistence=0.5,
        lacunarity=2
    )

    terrain = large + 0.15 * small
    return terrain[:256,:256]

def save_terrain(terrain:np.ndarray,
                 filename_prefix:str):
    size:int=4096
    center:int=1024
    min_grey:float=2**16/8
    zero_grey:float=2**16/4
    max_grey:float=7*(2**16/8)
    tmax,tmin=terrain.max(),terrain.min()
    def f(t):
        if t<=0:
            x=-t/tmin
        else:
            x=t/tmax
        x=x**3
        if x<=0:
            return (x+1)*(zero_grey-min_grey)+min_grey
        return x*(max_grey-zero_grey)+zero_grey
    f=np.vectorize(f)
    wm=np.round(f(terrain)).astype(np.uint16)
    wm=cv2.resize(wm,(size,size),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"{filename_prefix}_wm.png",wm)
    hm=wm[(size//2-center//2):(size//2+center//2),
       (size//2-center//2):(size//2+center//2)]
    hm=cv2.resize(hm,(size,size),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"{filename_prefix}_hm.png",hm)

if __name__=="__main__":
    for it in range(10):
        print("creating",it,"...")
        terrain=create_terrain()
        save_terrain(terrain,f"example_maps/{it}")
        plt.figure(figsize=(8,8))
        plt.imshow(terrain,cmap="terrain")
        plt.colorbar()
        plt.savefig(f"example_maps/{it}.png",dpi=300)