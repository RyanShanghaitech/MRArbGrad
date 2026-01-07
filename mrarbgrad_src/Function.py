from numpy import *
from matplotlib.pyplot import *
from numpy.typing import NDArray
from typing import Callable
import mrarbgrad.ext as ext
from .Utility import *

goldang = getGoldang()

def calGrad4ExFunc\
(
    fov: float64 = 0.256,
    nPix: int64 = 256,
    
    sLim: float64 = 50 * 42.5756e6 * 0.256 / 256,
    gLim: float64 = 50e-3 * 42.5756e6 * 0.256 / 256,
    dt: float64 = 10e-6,
    
    getK: Callable|None = None,
    getDkDp: Callable|None = None,
    getD2kDp2: Callable|None = None,
    
    p0:float64 = 0e0, 
    p1:float64 = 1e0, 
) -> tuple[NDArray, NDArray]:
    '''
    :return: gradient waveform, corresponding parameter
    :rtype: tuple[NDArray, NDArray]
    '''
    return ext.calGrad4ExFunc\
    (
        float64(fov),
        int64(nPix),
        
        float64(sLim), 
        float64(gLim), 
        float64(dt), 
        
        getK,
        getDkDp,
        getD2kDp2,
        
        float64(p0),
        float64(p1), 
    )

def calGrad4ExSamp\
(
    fov: float64 = 0.256,
    nPix: int64 = 256,
    
    sLim: float64 = 50 * 42.5756e6 * 0.256 / 256,
    gLim: float64 = 50e-3 * 42.5756e6 * 0.256 / 256,
    dt: float64 = 10e-6,
    
    arrK: NDArray = np.empty((0,3)),
) -> tuple[NDArray, NDArray]:
    '''
    :return: gradient waveform, corresponding parameter
    :rtype: tuple[NDArray, NDArray]
    '''
    return ext.calGrad4ExSamp\
    (
        float64(fov),
        int64(nPix),
        
        float64(sLim), 
        float64(gLim), 
        float64(dt), 
        
        arrK
    )
    
def getG_Spiral\
(
    fov: float64 = 0.256,
    nPix: int64 = 256,
    
    sLim: float64 = 50 * 42.5756e6 * 0.256 / 256,
    gLim: float64 = 50e-3 * 42.5756e6 * 0.256 / 256,
    dt: float64 = 10e-6,
    
    nSlice: int64 = 1,
    kRhoPhi: float64 = 0.5 / (4 * pi)
) -> tuple[list[NDArray], list[NDArray]]:
    '''
    :return: list of trajectory start, list of gradient waveforms
    :rtype: tuple[list[NDArray], list[NDArray]]
    '''
    return ext.getG_Spiral\
    (
        float64(fov),
        int64(nPix),
        
        float64(sLim),
        float64(gLim),
        float64(dt),
        
        int64(nSlice),
        float64(kRhoPhi)
    )

def getG_VDSpiral\
(
    fov: float64 = 0.256,
    nPix: int64 = 256,
    
    sLim: float64 = 50 * 42.5756e6 * 0.256 / 256,
    gLim: float64 = 50e-3 * 42.5756e6 * 0.256 / 256,
    dt: float64 = 10e-6,

    nSlice: int64 = 1,
    kRhoPhi0: float64 = 0.5 / (8 * pi),
    kRhoPhi1: float64 = 0.5 / (2 * pi),
) -> tuple[list[NDArray], list[NDArray]]:
    '''
    :return: list of trajectory start, list of gradient waveforms
    :rtype: tuple[list[NDArray], list[NDArray]]
    '''
    return ext.getG_VDSpiral\
    (
        float64(fov),
        int64(nPix),
        
        float64(sLim),
        float64(gLim),
        float64(dt),
        
        int64(nSlice),
        float64(kRhoPhi0),
        float64(kRhoPhi1)
    )

def getG_VDSpiral_RT\
(
    fov: float64 = 0.256,
    nPix: int64 = 256,
    
    sLim: float64 = 50 * 42.5756e6 * 0.256 / 256,
    gLim: float64 = 50e-3 * 42.5756e6 * 0.256 / 256,
    dt: float64 = 10e-6,
    
    kRhoPhi0: float64 = 0.5 / (8 * pi),
    kRhoPhi1: float64 = 0.5 / (2 * pi),
    nAcq: int64 = 1000,
) -> tuple[list[NDArray], list[NDArray]]:
    '''
    :return: list of trajectory start, list of gradient waveforms
    :rtype: tuple[list[NDArray], list[NDArray]]
    '''
    return ext.getG_VDSpiral_RT\
    (
        float64(fov),
        int64(nPix),
        
        float64(sLim),
        float64(gLim),
        float64(dt),
        
        float64(kRhoPhi0),
        float64(kRhoPhi1),
        
        int64(nAcq)
    )

def getG_Rosette\
(
    fov: float64 = 0.256,
    nPix: int64 = 256,
    
    sLim: float64 = 50 * 42.5756e6 * 0.256 / 256,
    gLim: float64 = 50e-3 * 42.5756e6 * 0.256 / 256,
    dt: float64 = 10e-6,
    
    nSlice: int64 = 1,
    om1: float64 = 5*pi, 
    om2: float64 = 3*pi, 
    tMax: float64 = 1e0,
) -> tuple[list[NDArray], list[NDArray]]:
    '''
    :return: list of trajectory start, list of gradient waveforms
    :rtype: tuple[list[NDArray], list[NDArray]]
    '''
    return ext.getG_Rosette\
    (
        float64(fov),
        int64(nPix),
        
        float64(sLim),
        float64(gLim),
        float64(dt),
        
        int64(nSlice),
        float64(om1),
        float64(om2),
        float64(tMax)
    )

def getG_Rosette_Trad\
(
    fov: float64 = 0.256,
    nPix: int64 = 256,
    
    sLim: float64 = 50 * 42.5756e6 * 0.256 / 256,
    gLim: float64 = 50e-3 * 42.5756e6 * 0.256 / 256,
    dt: float64 = 10e-6,
    
    nSlice: int64 = 1,
    om1: float64 = 5*pi, 
    om2: float64 = 3*pi, 
    tMax: float64 = 1e0,
    tAcq: float64 = 2.523e-3,
) -> tuple[list[NDArray], list[NDArray]]:
    '''
    :return: list of trajectory start, list of gradient waveforms
    :rtype: tuple[list[NDArray], list[NDArray]]
    '''
    return ext.getG_Rosette_Trad\
    (
        float64(fov),
        int64(nPix),
        
        float64(sLim),
        float64(gLim),
        float64(dt),
        
        int64(nSlice),
        float64(om1),
        float64(om2),
        float64(tMax),
        float64(tAcq)
    )

def getG_Shell3d\
(
    fov: float64 = 0.256,
    nPix: int64 = 256,
    
    sLim: float64 = 50 * 42.5756e6 * 0.256 / 256,
    gLim: float64 = 50e-3 * 42.5756e6 * 0.256 / 256,
    dt: float64 = 10e-6,
    
    kRhoTht: float64 = 0.5 / (2 * pi),
) -> tuple[list[NDArray], list[NDArray]]:
    '''
    :return: list of trajectory start, list of gradient waveforms
    :rtype: tuple[list[NDArray], list[NDArray]]
    '''
    return ext.getG_Shell3d\
    (
        float64(fov),
        int64(nPix),
        
        float64(sLim),
        float64(gLim),
        float64(dt),
        
        float64(kRhoTht),
    )

def getG_Yarnball\
(
    fov: float64 = 0.256,
    nPix: int64 = 256,
    
    sLim: float64 = 50 * 42.5756e6 * 0.256 / 256,
    gLim: float64 = 50e-3 * 42.5756e6 * 0.256 / 256,
    dt: float64 = 10e-6,
    
    kRhoPhi: float64 = 0.5 / (2 * pi),
) -> tuple[list[NDArray], list[NDArray]]:
    '''
    :return: list of trajectory start, list of gradient waveforms
    :rtype: tuple[list[NDArray], list[NDArray]]
    '''
    return ext.getG_Yarnball\
    (
        float64(fov),
        int64(nPix),
        
        float64(sLim),
        float64(gLim),
        float64(dt),
        
        float64(kRhoPhi),
    )

def getG_Seiffert\
(
    fov: float64 = 0.256,
    nPix: int64 = 256,
    
    sLim: float64 = 50 * 42.5756e6 * 0.256 / 256,
    gLim: float64 = 50e-3 * 42.5756e6 * 0.256 / 256,
    dt: float64 = 10e-6,
    
    m: float64 = 0.07, 
    uMax: float64 = 20.0, 
) -> tuple[list[NDArray], list[NDArray]]:
    '''
    :return: list of trajectory start, list of gradient waveforms
    :rtype: tuple[list[NDArray], list[NDArray]]
    '''
    return ext.getG_Seiffert\
    (
        float64(fov),
        int64(nPix),
        
        float64(sLim),
        float64(gLim),
        float64(dt),
        
        float64(m),
        float64(uMax),
    )

def getG_Cones\
(
    fov: float64 = 0.256,
    nPix: int64 = 256,
    
    sLim: float64 = 50 * 42.5756e6 * 0.256 / 256,
    gLim: float64 = 50e-3 * 42.5756e6 * 0.256 / 256,
    dt: float64 = 10e-6,
    
    kRhoPhi: float64 = 0.5 / (4 * pi),
) -> tuple[list[NDArray], list[NDArray]]:
    '''
    :return: list of trajectory start, list of gradient waveforms
    :rtype: tuple[list[NDArray], list[NDArray]]
    '''
    return ext.getG_Cones\
    (
        float64(fov),
        int64(nPix),
        
        float64(sLim),
        float64(gLim),
        float64(dt),
        
        float64(kRhoPhi),
    )

def setSolverMtg(x): ext.setSolverMtg(x)
def setTrajRev(x): ext.setTrajRev(x)
def setGoldAng(x): ext.setGoldAng(x)
def setShuf(x): ext.setShuf(x)
def setMaxG0(x): ext.setMaxG0(x)
def setMaxG1(x): ext.setMaxG1(x)
def setExGEnd(x): ext.setExGEnd(x)
def setMagOverSamp(x): ext.setMagOverSamp(x)
def setMagSFS(x): ext.setMagSFS(x)
def setMagGradRep(x): ext.setMagGradRep(x)
def setMagTrajRep(x): ext.setMagTrajRep(x)
def setDbgPrint(x): ext.setDbgPrint(x)
