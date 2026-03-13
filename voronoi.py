import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

SIZE=1024
""" Grid size of the map """

N_SEEDS=1000
""" Number of seeds """

RNG=np.random.default_rng(42)
""" Random number generator"""

def sample_voronoi_seeds():
    """ Generate seeds; x,y positions and z heights follow different distributions """
    seeds=np.zeros((N_SEEDS,3))
    seeds[:,0]=RNG.random(N_SEEDS)
    seeds[:,1]=RNG.random(N_SEEDS)
    seeds[:,2]=RNG.gumbel(0.2,0.1,N_SEEDS)
    seeds[:,2]=np.clip(seeds[:,2],a_min=0,a_max=1)
    return seeds

def create_voronoi_map(seeds:np.ndarray):
    """ Create a voronoi map """
    map=np.empty((SIZE,SIZE),dtype=np.float32)
    tree=cKDTree(seeds[:,:2])
    for i in range(SIZE):
        xy=np.stack([
            np.array([(i+0.5)/SIZE]*SIZE),
            np.linspace(0.05,(SIZE+0.05)/(SIZE+1),num=SIZE)
        ],axis=1)
        dist,ind=tree.query(xy,k=1)
        map[:,i]=seeds[ind,2]
    return map

def save_map(map:np.ndarray):
    """ Save map, assuming heights are in range 0..1 """
    map=(map*(2**16-1)).astype(np.uint16)
    cv2.imwrite("map.png",map)

if __name__=="__main__":
    seeds=sample_voronoi_seeds()
    map=create_voronoi_map(seeds)
    save_map(map)