import numpy as np
from noise import Noise

def gauss(x,y,x0=0.5,y0=0.5,sigma=0.1,amp=1):
    xf=((x-x0)**2)/(2*sigma**2)
    yf=((y-y0)**2)/(2*sigma**2)
    return amp*np.exp(-1*(xf+yf))



rng=np.random.default_rng(42)

for it in range(5):
    seed=rng.integers(low=0,high=2**16)
    # create map
    map=Noise(seed=seed,size=4096,center=1024)
    # get the center pixel value
    x,y=map._size//2,map._size//2
    v0=map.value(x,y)
    # raise the center a little so its not the dead bottom
    def blend(v,x,y):
        g=gauss(x,y,amp=0.5)
        return g*v0+(1-g)*v
    map.glf(blend)
    # done
    map.to_png(f"map{it}_wm.png",
               f"map{it}_hm.png") # done
    print(map)


