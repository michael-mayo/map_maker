"""
Program to generate CS2 compatible maps that can be loaded into the map editor.

Sample usage (two random maps with height scaled to 75%):
 python map.py --seeds 45,3 --max_z 0.75
or with defaults (generates 3 maps with 100% height scaling):
 python map.py

For each seed, a world map and height map will be created in the current directory.
"""

import argparse
from typing import Tuple,List
import json
import numpy as np
import cv2
import opensimplex

class Noise:
    """ Immutable noise layer class"""

    def __init__(self,
                 seed:int=42,
                 amplitude:float=1.0,
                 frequency:float=1.0,
                 offset:Tuple[float,float]=(0.0,0.0),
                 size:int=4096):
        """ Constructor """
        self._seed=seed
        self._amplitude=amplitude
        self._frequency=frequency
        self._offset=offset
        self._size=size
        opensimplex.seed(seed)
        ix=np.linspace(0,1,num=size)*frequency+offset[0]
        iy=np.linspace(0,1,num=size)*frequency+offset[1]
        self._noise=amplitude*opensimplex.noise2array(ix,iy)

    def __str__(self):
        """ Stringifier """
        return json.dumps(self.stats(), indent=2)

    def stats(self):
        """ All accessors in one method """
        return {
            "seed":self._seed,
            "amplitude":self._amplitude,
            "frequency":self._frequency,
            "offset":self._offset,
            "size":self._size,
            "noise_min":self._noise.min(),
            "noise_max":self._noise.max(),
            "noise_mean":self._noise.mean()
        }

class NoiseStack:
    """ Immutable class representing a stack of noise layers used to build the map;
        includes some CS2-specific settings and post processing """

    def __init__(self,
                 seed:int=42,
                 octaves:int=8,
                 size:int=4096,
                 debug:bool=False):
        """ Constructor """
        self._seed=seed
        self._octaves=octaves
        self._size=size
        self._debug=debug
        self._rng=np.random.default_rng(self._seed)
        seeds=self._rng.integers(0,high=2**16,size=octaves)
        offsets=(self._rng.uniform(low=-1000,high=1000,size=octaves*2)
                 .reshape(octaves,2))
        layers=[Noise(seed=int(seeds[i]),
                      size=self._size,
                      amplitude=1.0/(2.3**i),
                      frequency=2**(i+2),
                      offset=tuple(offsets[i,:]))
                for i in range(octaves)]
        self._noise=layers[0]._noise
        for i in range(1,octaves):
            self._noise+=layers[i]._noise
        self._scale_heights_nonuniformly()
        self._flatten_central_area()
        self._noise_gradient_magnitude() # currently gradient/magnitudes not used

    def __str__(self):
        """ Stringifier """
        return json.dumps(self.stats(), indent=2)

    def stats(self):
        """ All accessors in one method """
        return {
            "seed": self._seed,
            "octaves":self._octaves,
            "size": self._size,
            "noise_min": self._noise.min(),
            "noise_max": self._noise.max(),
            "noise_mean": self._noise.mean(),
            "magnitude_min":self._magnitude.min(),
            "magnitude_max":self._magnitude.max(),
            "angle_min":self._angle.min(),
            "angle_max":self._angle.max(),
        }

    def to_cs2_png(self,
               min_z:float=0.0,
               max_z:float=1.0):
        """ Save noise 16 bit png world and height map images with height scaling;
            if the map is too steep try reducing max_z """
        stats=self.stats()
        result=((self._noise-stats["noise_min"])
                /(stats["noise_max"]-stats["noise_min"]))
        result*=(max_z-min_z)
        result+=min_z
        wm=(result * (2 ** 16 - 1)).astype(np.uint16)
        if self._size!=4096:
            wm=cv2.resize(wm,(4096,4096),interpolation=cv2.INTER_CUBIC)
        offset=4096//2-1024//2
        hm=wm[offset:(offset+1024),offset:(offset+1024)]
        hm=cv2.resize(hm,(4096,4096),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"wm_{self._seed}.png",wm)
        cv2.imwrite(f"hm_{self._seed}.png",hm)

    def _noise_gradient_magnitude(self):
        """ Helper method to compute the gradient and magnitude of the noise """
        noise=self._noise.astype(np.float32)
        gx = cv2.Sobel(noise, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(noise, cv2.CV_32F, 0, 1, ksize=3)
        magnitude,angle=cv2.cartToPolar(gx, gy, angleInDegrees=True)
        self._magnitude=magnitude.astype(np.float64)
        self._angle = angle.astype(np.float64)
        if self._debug:
            mag_img = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            ang_img = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX)
            mag_img = mag_img.astype(np.uint8)
            ang_img = ang_img.astype(np.uint8)
            cv2.imwrite("mag.png",mag_img)
            cv2.imwrite("angle.png",ang_img)

    def _flatten_central_area(self,size=512,sigma=512//4,strength=1):
        """ Applies a median height bias to the central size X size area,
            mainly for cs2 playability so we don't end up with a gradiant of 100 m """
        offset=4096//2-size//2
        central_area=self._noise[offset:(offset+size),offset:(offset+size)]
        median_height=np.percentile(central_area.flatten(),50)
        gaussian_1d=cv2.getGaussianKernel(ksize=4096,sigma=sigma)
        gaussian_2d=gaussian_1d@gaussian_1d.T
        gaussian_2d=cv2.normalize(gaussian_2d,None,0,strength,cv2.NORM_MINMAX)
        self._noise=((1-gaussian_2d)*self._noise)+(gaussian_2d*median_height)



    def _scale_heights_nonuniformly(self,min_z=0.2):
        """ Softens heights non uniformly across the map to reduce the appearance
            of repetition; does so by generating a random noise layer with low frequency
            and range 0.2-1.0, which is then multiplied against the current map;
            then subtracts min_z from the noise and abs it to to produce valley effects"""
        seed=self._rng.integers(0,high=2**16,size=1)[0]
        offsets=self._rng.uniform(low=-1000,high=1000,size=2)
        scaler=Noise(seed=int(seed),
                      size=self._size,
                      amplitude=1.0,
                      frequency=2,
                      offset=tuple(offsets))
        scaler._noise-=scaler._noise.min()
        scaler._noise/=scaler._noise.max()
        scaler._noise=scaler._noise*(1-min_z)+min_z
        self._noise=np.multiply(self._noise,scaler._noise**2)
        self._noise=np.abs(self._noise)


# launcher
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=lambda s: [int(x) for x in s.split(',')],
        default="1,2,3",
        help="Comma-separated list of positive integer seeds, e.g. 34,22,1919",
    )
    parser.add_argument(
        "--max_z",
        type=float,
        default=1.0,
        help="Height scaling, should be >0 and <=1, default 1"
    )
    args=parser.parse_args()
    for seed in args.seeds:
        print(f"generating map {seed}...")
        n=NoiseStack(seed=seed,debug=False)
        n.to_cs2_png(min_z=0.0,max_z=np.clip(args.max_z,a_min=0,a_max=1))
    print("done")
