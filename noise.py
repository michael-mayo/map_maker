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
    """ Immutable class representing a stack of noise layers """

    def __init__(self,
                 seed:int=42,
                 octaves:int=8,
                 size:int=4096,
                 debug:bool=False):
        self._seed=seed
        self._octaves=octaves
        self._size=size
        self._debug=debug
        self._rng=np.random.default_rng(self._seed)
        seeds=self._rng.integers(0,high=2**16,size=octaves)
        offsets=(self._rng.uniform(low=-1000,high=1000,size=octaves*2)
                 .reshape(octaves,2))
        self._layers=[Noise(seed=int(seeds[i]),
                            size=self._size,
                            amplitude=1.0/(2**i),
                            frequency=2**i,
                            offset=tuple(offsets[i,:]))
                      for i in range(octaves)]
        self._noise=self._layers[0]._noise.copy()
        for i in range(1,octaves):
            self._noise+=self._layers[i]._noise
        self._noise_gradient_magnitude()

    def __str__(self):
        """ Stringifier """
        result=""
        if self._debug:
            for l in self._layers:
                result+=f"{str(l)}\n"
        result+=json.dumps(self.stats(), indent=2)
        return result

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
        """ Save noise to a 16 bit png image """
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


# Tester code
for seed in [42,1,999999]:
    n=NoiseStack(seed=seed,debug=True)
    print(n)
    n.to_cs2_png(min_z=0.0,max_z=1.0)