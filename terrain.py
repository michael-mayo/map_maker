import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.ndimage import gaussian_filter
import cv2

def create_terrain(rng:np.random.Generator)->np.ndarray:
    size=512
    terrain=np.zeros((size,size))
    bases=rng.integers(2**16,size=3,dtype=np.uint16)
    for y in range(size):
        for x in range(size):
            nx=x/size
            ny=y/size
            wx=pnoise2(nx*3,ny*3,octaves=3,base=bases[0])
            wy=pnoise2(nx*3+5,ny*3+5,octaves=3,base=bases[1])
            nx2=nx+0.8*wx
            ny2=ny+0.8*wy
            elevation=pnoise2(nx2*2,ny2*2,octaves=6,base=bases[2])
            terrain[y,x]=elevation
    return gaussian_filter(terrain,4,mode="wrap")

def save_terrain(terrain:np.ndarray,filename_prefix:str):
    size=4096
    center=1024
    wm=(2**16-1)*(terrain-terrain.min())/(terrain.max()-terrain.min())
    wm=wm.astype(np.uint16)
    wm=cv2.resize(wm,(size,size),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"{filename_prefix}_wm.png",wm)
    offset=center//2
    hm=wm[(size-offset):(size+offset),(size-offset):(size+offset)]
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