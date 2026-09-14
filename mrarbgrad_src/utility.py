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
    unit conversion: Tesla/meter to Herz/pixel

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


def getK_Radial(nPix:int, nAx:int, nSpoke:int, enGoldAng:bool) -> List:
    """
    Generates a Radial sampling pattern.
    
    Args:
        nPix (int): Number of pixels.
        nAx (int): Number of dimensions.
        nSpoke (int): Number of radial spokes.
        enGoldAng (bool): If True, uses the golden angle for spoke spacing.
        
    Returns:
        List: A list of numpy arrays, each of shape (nPix, nAx), representing radial spokes.
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

def clip(lstArrGrad:list[NDArray]|NDArray, dt:float, sLim:float, gLim:float) -> list[NDArray]|NDArray:
    """
    Clip the slewrate and gradient amlitude of a list of gradient waveforms

    Args:
        lstArrGrad: list of gradient waveforms
        sLim: slewrate amplitude limit
        gLim: gradient amplitude limit

    Returns:
        Clipped gradient waveforms
    """
    if isinstance(lstArrGrad, ndarray): _lstArrGrad = [lstArrGrad.copy()]
    else: _lstArrGrad = lstArrGrad.copy()
    nPE = len(_lstArrGrad)
    for iPE in range(nPE):
        # slew-rate clipping
        arrGrad = _lstArrGrad[iPE]
        arrSlew = diff(arrGrad, 1, 0)/dt
        arrSlewNorm = norm(arrSlew, axis=-1)
        arrSlewNorm[where(arrSlewNorm==0)] += 1e-6
        arrSlewUnit = arrSlew/arrSlewNorm[:,newaxis]
        np.clip(arrSlewNorm, None, sLim, out=arrSlewNorm)
        arrSlew = arrSlewUnit*arrSlewNorm[:,newaxis]
        # gradient clipping
        arrGrad[:,:] = arrGrad[0,:]
        arrGrad[1:,:] += cumsum(arrSlew*dt, axis=0)
        arrGradNorm = norm(arrGrad, axis=-1)
        arrGradNorm[where(arrGradNorm==0)] += 1e-6
        arrGradUnit = arrGrad/arrGradNorm[:,newaxis]
        np.clip(arrGradNorm, None, gLim, out=arrGradNorm)
        arrGrad = arrGradUnit*arrGradNorm[:,newaxis]
        # 
        _lstArrGrad[iPE] = arrGrad
        
    if isinstance(lstArrGrad, ndarray):
        return _lstArrGrad[0]
    else:
        return _lstArrGrad

def integrate(arrGrad:NDArray, dtGrad:int|float, dtAdc:int|float, nShift:int|float=1.0) -> NDArray:
    """
    # description:
    integrate the gradient waveform to the trajectory

    # parameter
    `arrGrad`: array of gradient waveform
    `dtGrad`, `dtAdc`: temporal resolution of gradient system and ADC

    # return:
    interpolated trajectory
    """
    dtShift = nShift*dtAdc
    nGrad, nAx = arrGrad.shape
    nAdc = int(dtGrad/dtAdc)*(nGrad-1)
    arrGrad_Resamp = zeros([nAdc,nAx], dtype=float64)
    for iDim in range(nAx):
        arrGrad_Resamp[:,iDim] = interp(dtAdc*arange(nAdc)+dtShift, dtGrad*arange(nGrad), arrGrad[:,iDim])
    arrDk = zeros_like(arrGrad_Resamp)
    arrDk[0,:] = (arrGrad[0,:] + arrGrad_Resamp[0,:])*dtShift/2
    arrDk[1:,:] = (arrGrad_Resamp[:-1] + arrGrad_Resamp[1:])*dtAdc/2
    arrK = cumsum(arrDk,axis=0)
    return arrK

def delay(arrGrad:NDArray, tau:int|float) -> NDArray:
    """
    # description:
    delay the input gradient waveform by time constant tau

    # parameter
    `arrGrad`: array of single gradient waveform
    `tau`: time constant in RL circuit transfer function

    # return:
    delayed gradient waveform
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
    arrGrad_ov = fft.ifft(fft.fft(arrGrad_ov,axis=0)*fft.fft(arrImpResRL)[:,newaxis], axis=0).real
    
    # de-oversample
    arrGrad = arrGrad_ov[:nPt:ov,:]

    return arrGrad

def rotate(arr:NDArray, ang:float64, axis:int64) -> NDArray:
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

