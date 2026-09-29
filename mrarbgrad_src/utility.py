from numpy import *
from numpy.typing import *
from typing import *
import numpy as np
from numpy.linalg import norm
from itertools import product

goldrat = (1+sqrt(5))/2
goldang = (2*pi)/(1+goldrat)

def getGoldRat()->float: return goldrat
def getGoldAng()->float: return goldang

def tm2hzpx(x:NDArray|float, res:float, gamma:float=42.5756e6) -> NDArray|float:
    """
    unit conversion: Tesla/meter to Hertz/pixel

    Args:
        x (NDArray|float): to be converted
        res (float): spatial resolution in `m/px`
        gamma (float): gyromagnetic ratio in `Hz/T`

    Returns:
        NDArray|float: converted results
    """
    return x * (gamma*res)

def hzpx2tm(x:NDArray|float, res:float, gamma:float=42.5756e6) -> NDArray|float:
    """
    unit conversion: Herz/pixel to Tesla/meter 

    Args:
        x (NDArray|float): to be converted
        res (float): spatial resolution in `m/px`
        gamma (float): gyromagnetic ratio in `Hz/T`

    Returns:
        NDArray|float: converted results
    """
    return x / (gamma*res)

def getK_Cartesian(nPix:int, nAx:int) -> List[NDArray]:
    """
    Generates a Cartesian sampling pattern.
    
    Args:
        nPix (int): Number of pixels.
        nAx (int): Number of dimensions.
        
    Returns:
        List[NDArray]: List of trajectories in (nK,nAx).
    """
    arrK1D = linspace(-0.5, 0.5, nPix, endpoint=False)
    lstArrK:list[NDArray] = []
    
    for arrKyKz in product(arrK1D, repeat=nAx-1):
        arrK = empty((nPix,nAx), dtype=float32)
        arrK[:,0] = arrK1D
        arrK[:,1:] = arrKyKz
        lstArrK.append(arrK)
        
    return lstArrK


def getK_Radial(nPix:int, nAx:int, nSpoke:int, enGoldAng:bool) -> List[NDArray]:
    """
    Generates a Radial sampling pattern.
    
    Args:
        nPix (int): Number of pixels.
        nAx (int): Number of dimensions.
        nSpoke (int): Number of radial spokes.
        enGoldAng (bool): If True, uses the golden angle for spoke spacing.
        
    Returns:
        List[NDArray]: A list of numpy arrays, each of shape (nPix, nAx), representing radial spokes.
    """
    if nAx!=2: raise NotImplementedError("nAx!=2")
    lstArrK = []
    arrK1D = linspace(-0.5, 0.5, nPix, endpoint=False)
    
    for iSpoke in range(nSpoke):
        if enGoldAng: tht = iSpoke * goldang
        else: tht = iSpoke * (pi / nSpoke)
        arrK = empty((nPix,nAx), dtype=float32)
        arrK[:,0] = arrK1D * cos(tht)
        arrK[:,1] = arrK1D * sin(tht)
        lstArrK.append(arrK)
        
    return lstArrK

def clip(arrGrad:NDArray, dt:float, sLim:float, gLim:float) -> NDArray:
    """
    Clip the slewrate and gradient amlitude of a gradient waveform.

    Args:
        arrGrad (NDArray): gradient waveform array
        dt (float): time resolution
        sLim (float): slewrate amplitude limit
        gLim (float): gradient amplitude limit

    Returns:
        NDArray: Clipped gradient waveforms
    """
    # slew-rate clipping
    arrSlew = diff(arrGrad, 1, 0)/dt
    arrSlewNorm = norm(arrSlew, axis=-1)
    arrSlewNorm[where(arrSlewNorm==0)] += 1e-6
    arrSlewUnit = arrSlew/arrSlewNorm[:,newaxis]
    np.clip(arrSlewNorm, None, sLim, out=arrSlewNorm)
    arrSlew = arrSlewUnit*arrSlewNorm[:,newaxis]
    # gradient clipping
    arrGrad = arrGrad.copy()
    arrGrad[:,:] = arrGrad[0,:]
    arrGrad[1:,:] += cumsum(arrSlew*dt, axis=0)
    arrGradNorm = norm(arrGrad, axis=-1)
    arrGradNorm[where(arrGradNorm==0)] += 1e-6
    arrGradUnit = arrGrad/arrGradNorm[:,newaxis]
    np.clip(arrGradNorm, None, gLim, out=arrGradNorm)
    arrGrad = arrGradUnit*arrGradNorm[:,newaxis]
    
    return arrGrad

def integrate(arrGrad:NDArray, dtGrad:int|float, dtAdc:int|float, nShift:int|float=1.0) -> NDArray:
    """
    integrate the gradient waveform to the trajectory

    Args:
        arrGrad (NDArray): array of gradient waveform
        dtGrad (int|float): temporal resolution of the gradient
        dtAdc (int|float): temporal resolution of the ADC
        nShift (int|float): at what position does ADC signal to be evaluated

    Returns:
        interpolated trajectory
    """
    dtShift = nShift*dtAdc
    nGrad, nAx = arrGrad.shape
    nAdc = (dtGrad/dtAdc)*(nGrad-1)
    nAdc = int(nAdc)
    arrGrad_Resamp = zeros([nAdc,nAx], dtype=float64)
    for iDim in range(nAx):
        arrGrad_Resamp[:,iDim] = interp(dtAdc*arange(nAdc)+dtShift, dtGrad*arange(nGrad), arrGrad[:,iDim])
    arrDk = zeros_like(arrGrad_Resamp)
    arrDk[0,:] = (arrGrad[0,:] + arrGrad_Resamp[0,:])*dtShift/2
    arrDk[1:,:] = (arrGrad_Resamp[:-1] + arrGrad_Resamp[1:])*dtAdc/2
    arrK = cumsum(arrDk,axis=0)
    return arrK

def delay(arrGrad:NDArray, tau:float) -> NDArray:
    """
    delay the input gradient waveform by time constant tau

    Args:
        arrGrad (NDArray): array of single gradient waveform
        tau (float): time constant in RL transfer function over gradeint raster time

    Returns:
        NDArray: delayed gradient waveform
    """
    assert arrGrad.ndim == 2, "only single gradient waveform is supported."
    if tau == 0: return arrGrad.copy() # avoid divided-by-0 later
    nPt, nAx = arrGrad.shape

    # perform oversample to get better impluse response profile
    ov = np.clip(10/tau, 1, 1e3).astype(int64) # the smaller the ov, the bigger oversampling is needed
    arrGrad_ov = zeros([nPt*ov,nAx], dtype=arrGrad.dtype)
    for iAx in range(nAx):
        arrGrad_ov[:,iAx] = interp(linspace(0,nPt,nPt*ov,False), linspace(0,nPt,nPt,False), arrGrad[:,iAx]) # oversample
    nPt *= ov
    tau *= ov

    # derive impluse response of RL circuit
    arrGrad_Pad = zeros_like(arrGrad_ov)
    arrGrad_Pad[:nPt//2,:] = arrGrad_ov[-1:,:]
    arrGrad_Pad[nPt//2:,:] = arrGrad_ov[:1,:]
    arrGrad_ov = concatenate([arrGrad_ov, arrGrad_Pad], axis=0)
    arrT = linspace(0,2*nPt,2*nPt,False) + 0.5
    arrImpResRL = (1/tau)*exp(-arrT/tau)
    if abs(arrImpResRL.sum() - 1) > 1e-2: raise ValueError(f"arrImpResRL.sum() = {arrImpResRL.sum():.2f} (supposed to be 1) (tau too small or too large)")
    
    # perform convolution between input waveform and impulse response
    arrGrad_ov = fft.ifft(fft.fft(arrGrad_ov,axis=0) * fft.fft(arrImpResRL)[:,newaxis], axis=0).real
    
    # deoversample
    arrGrad = arrGrad_ov[:nPt:ov,:]

    return arrGrad

def rotate(arr:NDArray, ang:float64, axis:int64) -> NDArray:
    r"""
    Apply rotation matrix to an (N,3) array.

    Args:
        arr (NDArray): array to be rotated, shape: (N,3)
        ang (NDArray): rotation angle in radian
        axis (NDArray): along which axis to rotate

    Returns:
        NDArray: rotated `arr`
    """
    if axis==0: # x
        matRot = array([
            [1, 0, 0],
            [0, cos(ang), -sin(ang)],
            [0, sin(ang), cos(ang)],
        ], dtype=float64)
    elif axis==1: # y
        matRot = array([
            [cos(ang), 0, sin(ang)],
            [0, 1, 0],
            [-sin(ang), 0, cos(ang)]
        ], dtype=float64)
    elif axis==2: # z
        matRot = array([
            [cos(ang), -sin(ang), 0],
            [sin(ang), cos(ang), 0],
            [0, 0, 1]
        ], dtype=float64)
    else:
        raise ValueError("axis should be 0, 1, 2 (denotes for x, y, z)")
    
    return arr@matRot.T

def rand3d(i:int|NDArray, nAx:int=3, kx=sqrt(2), ky=sqrt(3), kz=sqrt(7)) -> NDArray:
    return (hstack if size(i)==1 else vstack)\
    ([
        (i**1 * 1/(1+kx))%1,
        (i**2 * 1/(1+ky))%1,
        (i**3 * 1/(1+kz))%1
    ][:nAx]).T
    
def genPermTab(n:int) -> list[int]:
    inc = around(n*(goldrat-1)).astype(int64)
    while gcd(inc,n)!=1: inc-=1
    lstIdx = []
    for i in range(n):
        lstIdx.append(i*inc%n)
    return lstIdx

