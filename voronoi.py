import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

SIZE=1024
""" Grid size of the map """

SEQS=[(50,0.5),(100,0.25),(200,0.125),(400,0.065),(800,0.0325)]
""" Sequences of #vornoi seeds,scale pairs  """

RNG_SEED=42
RNG=np.random.default_rng(RNG_SEED)
""" Random number generator"""

JITTER=0.003
""" Amount of seed jitter when constructing voronoi map"""

def sample_voronoi_seeds(n_seeds:int):
    """ Generate seeds; x,y positions and z heights follow different distributions """
    seeds=np.zeros((n_seeds,3))
    seeds[:,0]=RNG.random(n_seeds)
    seeds[:,1]=RNG.random(n_seeds)
    seeds[:,2]=RNG.gumbel(0.2,0.1,n_seeds)
    seeds[:,2]=np.clip(seeds[:,2],a_min=0,a_max=1)
    return seeds

def create_voronoi_map(seeds:np.ndarray):
    """ Create a jittered voronoi map """
    map=np.empty((SIZE,SIZE),dtype=np.float32)
    tree = cKDTree(seeds[:, :2])
    for i in range(SIZE):
        xy=np.stack([
            np.array([(i+0.5)/SIZE]*SIZE),
            np.linspace(0.05,(SIZE+0.05)/(SIZE+1),num=SIZE)
        ],axis=1)
        dist,ind=tree.query(xy,k=1)
        map[:,i]=seeds[ind,2]
        if JITTER>0 and i<SIZE-1:
            seeds[:,:2]+=JITTER*(RNG.random(seeds[:,:2].shape)-0.5)
            seeds[:,:2]=np.clip(seeds[:,:2],a_min=0.0,a_max=1.0)
            tree=cKDTree(seeds[:,:2])
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
    map=np.zeros((SIZE,SIZE),dtype=np.float32)
    for num_voronoi_seeds,scale in SEQS:
        voronoi_seeds=sample_voronoi_seeds(num_voronoi_seeds)
        map+=scale*create_voronoi_map(voronoi_seeds)
    map=gaussian_filter(map,10)
    map=(map-map.min())/(map.max()-map.min())
    save_map(map)