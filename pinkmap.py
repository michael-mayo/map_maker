
import numpy as np
import cv2

from typing import Tuple
from scipy.ndimage import gaussian_filter

RNG=np.random.Generator
""" Random number generator"""

def pink_noise_2d(shape:Tuple[int,int],
                  beta:float=1.0):
    """ Generate 2D pink noise when beta=1.0 """
    h,w=shape
    fy=np.fft.fftfreq(h)[:,None]
    fx=np.fft.fftfreq(w)[None,:]
    f=np.sqrt(fx*fx+fy*fy)
    f[0,0]=1.0
    spectrum=RNG.normal(size=(h,w))+1j*RNG.normal(size=(h,w))
    spectrum/=np.power(f,beta/2.0)
    field=np.fft.ifft2(spectrum).real
    field-=field.min()
    field/=field.max() + 1e-12
    return field

def bilinear_sample(img:np.ndarray,
                    x:float,
                    y:float)->float:
    """ Sample pixel value given non-rounded float coordinates """
    h,w=img.shape
    x0=np.floor(x).astype(int) % w
    x1=(x0+1)%w
    y0=np.floor(y).astype(int) % h
    y1=(y0+1)%h
    sx=x-np.floor(x)
    sy=y-np.floor(y)
    v00=img[y0,x0]
    v10=img[y0,x1]
    v01=img[y1,x0]
    v11=img[y1,x1]
    return (
        (1-sx)*(1-sy)*v00 +
        sx*(1-sy)*v10 +
        (1-sx)*sy*v01 +
        sx*sy*v11
    )

def domain_warped_pink_noise(shape:Tuple[int,int]=(256, 256),
                             base_beta:float=4.0,
                             warp_beta:float=1.0,
                             warp_strength:float=60.0):
    """ Generate domain warped pink noise """
    h,w=shape
    base=pink_noise_2d(shape,beta=base_beta)
    warp_x=pink_noise_2d(shape,beta=warp_beta)
    warp_y=pink_noise_2d(shape,beta=warp_beta)
    warp_x=2.0*warp_x-1.0
    warp_y=2.0*warp_y-1.0
    yy,xx=np.mgrid[0:h,0:w]
    xw=xx+warp_strength*warp_x
    yw=yy+warp_strength*warp_y
    warped=bilinear_sample(base,xw,yw)
    warped-=warped.min()
    warped/=warped.max()+1e-12
    return warped


def create_map(
        seed:int,
        shape:Tuple[int,int]=(256,256),
        smoothing_sigma:float=4)->np.ndarray:
    """ Create a map """
    global RNG
    RNG=np.random.default_rng(seed)
    result=domain_warped_pink_noise(shape)
    result=gaussian_filter(result,smoothing_sigma)
    
    # TODO: shift the height of the starting square down to 0.3
    # TODO: flatten the starting square; height map is divided into 23x23 squares
    # TODO: and the central 3x3 are the game start squares

    return result


def save_map(filename_suffix:str,terrain:np.ndarray):
    """ Save the map as 16 bit png """
    tmin,tmax=terrain.min(),terrain.max()
    wm=((2**16-1)*(terrain-tmin)/(tmax-tmin)).astype(np.uint16)
    wm=cv2.resize(wm,(4096,4096),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"example_maps/wm_{filename_suffix}",wm)
    hm=wm[(4096//2-1024//2):(4096//2+1024//2),
          (4096//2-1024//2):(4096//2+1024//2)]
    hm=cv2.resize(hm,(4096,4096),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"example_maps/hm_{filename_suffix}.png", hm)

if __name__=="__main__":
    for seed in [42,22,111]:
        save_map(
            f"{seed}.png",
            create_map(seed,shape=(512,512)))
