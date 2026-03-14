import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

SIZE=1024
""" Grid size of the map """

N_SEEDS=100
""" Number of vornoi seeds """

RNG_SEED=42
RNG=np.random.default_rng(RNG_SEED)
""" Random number generator"""

K=10
""" Number of nearest voronoi seeds to consider """

def sample_voronoi_seeds(n_seeds:int):
    """ Generate seeds; x,y positions and z heights follow different distributions """
    seeds=np.zeros((n_seeds,3))
    seeds[:,0]=RNG.random(n_seeds)
    seeds[:,1]=RNG.random(n_seeds)
    seeds[:,2]=RNG.random(n_seeds)
    return seeds

def create_voronoi_map(seeds:np.ndarray):
    """ Create a voronoi map """
    map=np.empty((SIZE,SIZE),dtype=np.float32)
    tree = cKDTree(seeds[:, :2])
    for i in range(SIZE):
        xy=np.stack([
            np.array([(i+0.5)/SIZE]*SIZE),
            np.linspace(0.05,(SIZE+0.05)/(SIZE+1),num=SIZE)
        ],axis=1)
        dist,ind=tree.query(xy,k=K)
        heights=(np.vectorize(
            lambda seed_ind:seeds[seed_ind,2])
            (ind))
        map[:,i]=heights.mean(axis=1)
    return map


def save_map(map:np.ndarray):
    """ Save map, assuming heights are in range 0..1 """
    wm=(map*(2**16-1)).astype(np.uint16)
    wm=cv2.resize(wm,(4096,4096),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"wm_{RNG_SEED}.png",wm)
    hm=wm[(4096//2-1024//2):(4096//2+1024//2),
          (4096//2-1024//2):(4096//2+1024//2)]
    hm=cv2.resize(hm,(4096,4096),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"hm_{RNG_SEED}.png", hm)

if __name__=="__main__":
    voronoi_seeds=sample_voronoi_seeds(N_SEEDS)
    map=create_voronoi_map(voronoi_seeds)
    map=gaussian_filter(map,7)
    map=(map-map.min())/(map.max()-map.min())
    save_map(map)