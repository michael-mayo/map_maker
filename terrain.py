import numpy as np
import matplotlib.pyplot as plt
from noise import snoise2
from scipy.ndimage import gaussian_filter
import cv2
from typing import List,Tuple

def create_terrain(rng:np.random.Generator)->np.ndarray:
    bases=list(map(int,rng.integers(2**16-1,size=4,dtype=np.uint16)))
    size = 1024
    terrain = np.zeros((size, size), dtype=np.float32)
    warp_freq = 1.2
    warp_amp = 0.008
    large_freq = 0.7
    detail_freq = 4.5
    for y in range(size):
        ny = (y + 0.5) / size
        for x in range(size):
            nx = (x + 0.5) / size
            wx = snoise2(nx * warp_freq + 17.3, ny * warp_freq - 8.1, octaves=2,base=bases[0])
            wy = snoise2(nx * warp_freq - 23.7, ny * warp_freq + 11.4, octaves=2,base=bases[1])
            nx2 = nx + warp_amp * wx
            ny2 = ny + warp_amp * wy
            large = snoise2(nx2 * large_freq + 31.1, ny2 * large_freq - 12.8, octaves=2,base=bases[2])
            small = snoise2(nx2 * detail_freq - 7.4, ny2 * detail_freq + 22.6, octaves=4,base=bases[3])
            terrain[y, x] = large + 0.2 * small
    return terrain

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
    rng=np.random.default_rng(42)
    seeds=rng.integers(2**16,size=10,dtype=np.uint16)
    for seed in seeds:
        print("creating",seed,"...")
        terrain=create_terrain(rng)
        save_terrain(terrain,f"example_maps/{seed}")
        plt.figure(figsize=(8,8))
        plt.imshow(terrain,cmap="terrain")
        plt.colorbar()
        plt.savefig(f"example_maps/{seed}.png",dpi=300)