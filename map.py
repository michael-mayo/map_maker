import argparse
import numpy as np
import cv2
from multiprocessing import Pool
from noise import Noise


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="randon number seed, a positive int")
parser.add_argument("--it", type=int, default=3, help="number of iterations")
parser.add_argument("--num_drops", type=int, default=0, help="number of water drop simulations per map")
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
        nhoodv = [map.value(*n)
                  if n[0] >= 0 and n[0] < 4096 and n[1] >= 0 and n[1] < 4096 else np.nan
                  for n in nhood]
        i = np.argmin(nhoodv)
        if nhoodv[i] is not None and nhoodv[i] < cv:
            c = nhood[i]
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
    if args.num_drops>0:
        print("...starting drop sims")
        water = np.zeros((4096, 4096), dtype=np.float32)
        cc=rng.integers(low=1,high=4096,size=(args.num_drops,2))
        for i in range(args.num_drops):
            sim_drop(i)
            if (i+1)%100==0:
                print(".",end="")
        if args.num_drops>=100:
            print()
        water-=water.min()
        water/=water.max()
        water=water**0.5
        print("...drop sims complete")
    print("saving map files...")
    map.to_png(f"map{it}_wm.png",f"map{it}_hm.png")
    if args.num_drops > 0:
        cv2.imwrite(f"map{it}_water.png",(water*(2**16-1)).astype(np.uint16))




