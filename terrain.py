import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.ndimage import gaussian_filter
import cv2
from typing import List,Tuple

def create_terrain(rng:np.random.Generator)->np.ndarray:
    size=1024
    terrain=np.zeros((size,size))
    bases=rng.integers(2**16,size=3,dtype=np.uint16)
    for y in range(size):
        for x in range(size):
            nx=x/size
            ny=y/size
            wx=pnoise2(nx*3,ny*3,octaves=6,base=bases[0])
            wy=pnoise2(nx*3+5,ny*3+5,octaves=6,base=bases[1])
            nx2=nx+0.1*wx
            ny2=ny+0.1*wy
            elevation=pnoise2(nx2*3,ny2*3,octaves=6,base=bases[2])
            terrain[y,x]=elevation
    #terrain=gaussian_filter(terrain,4,mode="wrap")
    return terrain

def save_terrain(terrain:np.ndarray,
                 filename_prefix:str):
    size:int=4096
    center:int=1024
    min_grey:float=2**16/8
    zero_grey:float=3**16/4
    max_grey:float=7*2**16/8
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
    seeds=rng.integers(2**16,size=4,dtype=np.uint16)
    for seed in seeds:
        print("creating",seed,"...")
        terrain=create_terrain(rng)
        save_terrain(terrain,f"example_maps/{seed}")
        plt.figure(figsize=(8,8))
        plt.imshow(terrain,cmap="terrain")
        plt.colorbar()
        plt.savefig(f"example_maps/{seed}.png",dpi=300)