import argparse
import numpy as np
import cv2
from multiprocessing import Pool
from noise import Noise


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="randon number seed, a positive int")
parser.add_argument("--it", type=int, default=3, help="number of iterations")
parser.add_argument("--drops", type=int, default=0, help="number of water drop simulations per map")
args = parser.parse_args()

rng=np.random.default_rng(args.seed)


def sim_drop(i):
    global cc
    global water
    global map
    c = cc[i, :]
    while True:
        cv = map.value(*c)
        cx, cy = c[0], c[1]
        nhood = [(cx - 1, cy - 1), (cx, cy - 1), (cx + 1, cy - 1),
                 (cx - 1, cy), (cx + 1, cy),
                 (cx - 1, cy + 1), (cx, cy + 1), (cx + 1, cy + 1)]
        np.random.shuffle(nhood)
        nhoodv = [map.value(*n)
                  if n[0] >= 0 and n[0] < 4096 and n[1] >= 0 and n[1] < 4096 else np.nan
                  for n in nhood]
        nhoodv_sorted_idx=np.argsort(nhoodv)
        lowest_idx=nhoodv_sorted_idx[0]
        candidate_idx=nhoodv_sorted_idx[np.random.choice([0,1,2])]
        if nhoodv[candidate_idx] is not None and nhoodv[lowest_idx] < cv:
            c = nhood[candidate_idx]
            water[*c] += 0.01
        else:
            break


for it in range(args.it):
    print(f"generating map {it}...")
    seed=rng.integers(low=0,high=2**16)
    map=Noise(seed=seed,
              size=4096,
              center=1024,
              octaves=8,
              base_freq=rng.uniform()*1.5+0.5)
    map.center(method="mean", standardise=True)
    map.center(method="center_pixel",standardise=False)
    map.ptf(lambda v: v**3 if v>=0 else v)
    print(f"...map generated with following params:\n...{str(map)}")
    if args.drops>0:
        print("...starting drop sims")
        water = np.zeros((4096, 4096), dtype=np.float32)
        cc=rng.integers(low=1,high=4096,size=(args.drops,2))
        for i in range(args.drops):
            sim_drop(i)
            if (i+1)%100==0:
                print(".",end="",flush=True)
        if args.drops>=100:
            print()
        water-=water.min()
        water/=water.max()
        water=(water * (2 ** 16 - 1)).astype(np.uint16)
        water=cv2.dilate(water,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),
                         iterations=1)
        water=np.power(water,0.5)
        for _ in range(4):
            water=cv2.GaussianBlur(water,(17,17),0)
        map_max,map_min=map._noise.max(),map._noise.min()
        map_range=map_max-map_min
        map._noise-=(0.04*map_range*water.astype(np.float32)/water.max())
        #map._noise=cv2.GaussianBlur(map._noise,(3,3),0)
        print("...drop sims complete")
    print("saving map files...")
    map.to_png(f"map{it}_wm.png",f"map{it}_hm.png")
    if args.drops > 0:
        cv2.imwrite(f"_map{it}_water.png",water)




