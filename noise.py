import numpy as np
import cv2
import opensimplex

class Noise:
    
    def __init__(self,
                 size:int=4096,
                 seed:int=42,
                 octaves:int=8,
                 lacunarity:float=2.0,
                 persistence:float=0.5,
                 base_freq:float=1.0,
                 center:int=1024):
        """ Constructor """
        self._size=size
        self._seed=seed
        self._octaves=octaves
        self._lacunarity=lacunarity
        self._persistence=persistence
        self._base_freq=base_freq
        self._center=center
        with np.errstate(over="ignore"):
            opensimplex.seed(seed)
        ix=np.linspace(0,1,num=size)
        iy=np.linspace(0,1,num=size)
        rng = np.random.default_rng(seed)
        self._noise=np.zeros((size,size),dtype=np.float32)
        for j in range(octaves):
            freq=base_freq*(lacunarity**j)
            amp=(persistence**j)
            ox,oy=rng.uniform(0,10000,size=2)
            self._noise+=amp*opensimplex.noise2array(ix*freq+ox,iy*freq+oy)

    def __str__(self)->str:
        """ Stringifier """
        params=[f"size={self._size}"]
        params+=[f"seed={self._seed}"]
        params+=[f"octaves={self._octaves}"]
        params+=[f"lacunarity={self._lacunarity}"]
        params+=[f"persistence={self._persistence}"]
        params+=[f"base_freq={self._base_freq}"]
        params+=[f"center={self._center}"]
        params+=[f"range={self._noise.min()},{self._noise.max()}"]
        return f"Noise({','.join(map(str,params))})"

    def value(self,x,y):
        return self._noise[x,y]


    def add_k(self,k:float):
        self._noise+=k
        return self

    def add(self,other):
        self._noise+=other._noise
        return self

    def mlt_k(self,k:float):
        self._noise*=k
        return self

    def mlt(self,other):
        self._noise=np.multiply(self._noise,other._noise)
        return self

    def max(self,other):
        self._noise=np.maximum(self._noise,other._noise)
        return self

    def smooth(self,ksize:int=11):
        self._noise=cv2.GaussianBlur(self._noise,(ksize,ksize),0)
        return self

    def center(self,standardise=False):
        """ Center the noise to mean 0 and optionally scale it to std dev 1"""
        self._noise-=self._noise.mean()
        if standardise:
            self._noise/=self._noise.std()
        return self

    def ptf(self,f):
        """ Apply a point function to each cell in the noise array. Function f
            should take a single float param, return a single float, and not
            be an already vectorized function """
        self._noise=(np.vectorize(f))(self._noise)
        return self

    def glf(self,f):
        """ Apply a global function to each cell in the noise array.
            The function f should take three float params being value,x,y, and
            return a single value. The x,y coordinates are scaled 0..1 so the center
            pixel will have coordinate (0.5,0.5). The function f should not be already vectorised """
        ix=np.linspace(0,1,num=self._size)
        iy=np.linspace(0,1,num=self._size)
        xx,yy=np.meshgrid(ix,iy,indexing="xy")
        self._noise = (np.vectorize(f))(self._noise,xx,yy)
        return self

    def to_png(self,
               filename,
               centercrop_filename=None,
               max_intensity=2**16-1):
        """ Generate a 16 bit PNG image from the noise, and optionally a resized center crop
            for the City Skylines 2 map editor """
        min,max=self._noise.min(),self._noise.max()
        img=max_intensity*(self._noise-min)/(max-min)
        img=img.astype(np.uint16)
        cv2.imwrite(filename,img)
        if centercrop_filename is not None:
            offset=self._size//2-self._center//2
            img=img[offset:(offset+self._center),offset:(offset+self._center)]
            img=cv2.resize(img,(self._size,self._size),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(centercrop_filename,img)
        return self