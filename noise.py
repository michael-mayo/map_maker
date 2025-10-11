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
                 center=1024):
        self._size=size
        self._seed=seed
        self._octaves=octaves
        self._lacunarity=lacunarity
        self._persistence=persistence
        self._base_freq=base_freq
        self._center=1024
        opensimplex.seed(seed)
        ix=np.linspace(0,1,num=size)
        iy=np.linspace(0,1,num=size)
        rng = np.random.default_rng(seed)
        self._noise=np.zeros((size,size),dtype=np.float32)
        for j in range(octaves):
            freq=base_freq*(lacunarity**j)
            amp=(persistence**j)
            ox,oy=rng.uniform(0,10000,size=2)
            self._noise+=amp*opensimplex.noise2array(ix*freq+ox,ix*freq+oy)

    def __str__(self)->str:
        params=[f"size={self._size}"]
        params+=[f"seed={self._seed}"]
        params+=[f"octaves={self._octaves}"]
        params+=[f"lacunarity={self._lacunarity}"]
        params+=[f"persistence={self._persistence}"]
        params+=[f"base_freq={self._base_freq}"]
        params+=[f"center={self._center}"]
        return f"Noise({','.join(map(str,params))})"

    def center(self,standardise=False):
        self._noise-=self._noise.mean()
        if standardise:
            self._noise/=self._noise.std()
        return self

    def ptf(self,f):
        self._noise=(np.vectorize(f))(self._noise)
        return self

    def flatten_center(self,size=512,sigma=512//4,strength=1):
        """ Applies a median height bias to the central size X size area,
            mainly for cs2 playability so we don't end up with a gradiant of 100 m """
        offset=self._size//2-size//2
        central_area=self._noise[offset:(offset+size),offset:(offset+size)]
        median_height=np.percentile(central_area.flatten(),50)
        gaussian_1d=cv2.getGaussianKernel(ksize=4096,sigma=sigma)
        gaussian_2d=gaussian_1d@gaussian_1d.T
        gaussian_2d=cv2.normalize(gaussian_2d,None,0,strength,cv2.NORM_MINMAX)
        self._noise=((1-gaussian_2d)*self._noise)+(gaussian_2d*median_height)
        return self

    def to_png(self,
               filename,
               centercrop_filename=None,
               max_intensity=2**16-1):
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