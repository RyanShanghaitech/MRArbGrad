from numpy import *
from matplotlib.pyplot import *
from numpy.typing import NDArray
from typing import Callable
import mrautograd.ext as ext

goldang = (3-sqrt(5))*pi

def calGrad\
(
    bIs3D: bool = False,
    dFov: float64 = 0.256,
    lNPix: int64 = 256,
    
    dSLim: float64 = 100 * 42.5756e6 * 0.25 / 256,
    dGLim: float64 = 120e-3 * 42.5756e6 * 0.25 / 256,
    dDt: float64 = 10e-6,
    
    getK: Callable|None = None,
    getDkDp: Callable|None = None,
    getD2kDp2: Callable|None = None,
    
    p0:float64 = 0e0, 
    p1:float64 = 1e0, 
) -> NDArray:
    return ext.calGrad\
    (
        bool(bIs3D),
        float64(dFov),
        int64(lNPix),
        
        float64(dSLim), 
        float64(dGLim), 
        float64(dDt), 
        
        getK,
        getDkDp,
        getD2kDp2,
        
        float64(p0),
        float64(p1), 
    )
    
def getG_Spiral\
(
    bIs3D: bool = False,
    dFov: float64 = 0.256,
    lNPix: int64 = 256,
    
    dSLim: float64 = 100 * 42.5756e6 * 0.25 / 256,
    dGLim: float64 = 120e-3 * 42.5756e6 * 0.25 / 256,
    dDt: float64 = 10e-6,
    
    dRhoPhi: float64 = 0.5 / (8 * pi)
) -> tuple[list[NDArray], list[NDArray]]:
    return ext.getG_Spiral\
    (
        bool(bIs3D),
        float64(dFov),
        int64(lNPix),
        
        float64(dSLim),
        float64(dGLim),
        float64(dDt),
        
        float64(dRhoPhi)
    )

def getG_VarDenSpiral\
(
    bIs3D: bool = False,
    dFov: float64 = 0.256,
    lNPix: int64 = 256,
    
    dSLim: float64 = 100 * 42.5756e6 * 0.25 / 256,
    dGLim: float64 = 120e-3 * 42.5756e6 * 0.25 / 256,
    dDt: float64 = 10e-6,
    
    dRhoPhi0: float64 = 0.5 / (16 * pi),
    dRhoPhi1: float64 = 0.5 / (4 * pi),
) -> tuple[list[NDArray], list[NDArray]]:
    return ext.getG_VarDenSpiral\
    (
        bool(bIs3D),
        float64(dFov),
        int64(lNPix),
        
        float64(dSLim),
        float64(dGLim),
        float64(dDt),
        
        float64(dRhoPhi0),
        float64(dRhoPhi1)
    )

def getG_Rosette\
(
    bIs3D: bool = False,
    dFov: float64 = 0.256,
    lNPix: int64 = 256,
    
    dSLim: float64 = 100 * 42.5756e6 * 0.25 / 256,
    dGLim: float64 = 120e-3 * 42.5756e6 * 0.25 / 256,
    dDt: float64 = 10e-6,
    
    dOm1: float64 = 10*pi, 
    dOm2: float64 = 8*pi, 
    dTmax: float64 = 1e0,
) -> tuple[list[NDArray], list[NDArray]]:
    return ext.getG_Rosette\
    (
        bool(bIs3D),
        float64(dFov),
        int64(lNPix),
        
        float64(dSLim),
        float64(dGLim),
        float64(dDt),
        
        float64(dOm1),
        float64(dOm2),
        float64(dTmax)
    )

def getG_Rosette_Trad\
(
    bIs3D: bool = False,
    dFov: float64 = 0.256,
    lNPix: int64 = 256,
    
    dSLim: float64 = 100 * 42.5756e6 * 0.25 / 256,
    dGLim: float64 = 120e-3 * 42.5756e6 * 0.25 / 256,
    dDt: float64 = 10e-6,
    
    dOm1: float64 = 10*pi, 
    dOm2: float64 = 8*pi, 
    dTmax: float64 = 1e0
) -> tuple[list[NDArray], list[NDArray]]:
    return ext.getG_Rosette_Trad\
    (
        bool(bIs3D),
        float64(dFov),
        int64(lNPix),
        
        float64(dSLim),
        float64(dGLim),
        float64(dDt),
        
        float64(dOm1),
        float64(dOm2),
        float64(dTmax)
    )

def getG_Shell3d\
(
    bIs3D: bool = False,
    dFov: float64 = 0.256,
    lNPix: int64 = 256,
    
    dSLim: float64 = 100 * 42.5756e6 * 0.25 / 256,
    dGLim: float64 = 120e-3 * 42.5756e6 * 0.25 / 256,
    dDt: float64 = 10e-6,
    
    dRhoTht: float64 = 0.5 / (2 * pi),
) -> tuple[list[NDArray], list[NDArray]]:
    return ext.getG_Shell3d\
    (
        bool(bIs3D),
        float64(dFov),
        int64(lNPix),
        
        float64(dSLim),
        float64(dGLim),
        float64(dDt),
        
        float64(dRhoTht),
    )

def getG_Yarnball\
(
    bIs3D: bool = False,
    dFov: float64 = 0.256,
    lNPix: int64 = 256,
    
    dSLim: float64 = 100 * 42.5756e6 * 0.25 / 256,
    dGLim: float64 = 120e-3 * 42.5756e6 * 0.25 / 256,
    dDt: float64 = 10e-6,
    
    dRhoPhi: float64 = 0.5 / (2 * pi),
) -> tuple[list[NDArray], list[NDArray]]:
    return ext.getG_Yarnball\
    (
        bool(bIs3D),
        float64(dFov),
        int64(lNPix),
        
        float64(dSLim),
        float64(dGLim),
        float64(dDt),
        
        float64(dRhoPhi),
    )

def getG_Seiffert\
(
    bIs3D: bool = False,
    dFov: float64 = 0.256,
    lNPix: int64 = 256,
    
    dSLim: float64 = 100 * 42.5756e6 * 0.25 / 256,
    dGLim: float64 = 120e-3 * 42.5756e6 * 0.25 / 256,
    dDt: float64 = 10e-6,
    
    dM: float64 = 0.07, 
    dUMax: float64 = 20.0, 
) -> tuple[list[NDArray], list[NDArray]]:
    return ext.getG_Seiffert\
    (
        bool(bIs3D),
        float64(dFov),
        int64(lNPix),
        
        float64(dSLim),
        float64(dGLim),
        float64(dDt),
        
        float64(dM),
        float64(dUMax),
    )

def getG_Cones\
(
    bIs3D: bool = False,
    dFov: float64 = 0.256,
    lNPix: int64 = 256,
    
    dSLim: float64 = 100 * 42.5756e6 * 0.25 / 256,
    dGLim: float64 = 120e-3 * 42.5756e6 * 0.25 / 256,
    dDt: float64 = 10e-6,
    
    dRhoPhi: float64 = 0.5 / (8 * pi),
) -> tuple[list[NDArray], list[NDArray]]:
    return ext.getG_Cones\
    (
        bool(bIs3D),
        float64(dFov),
        int64(lNPix),
        
        float64(dSLim),
        float64(dGLim),
        float64(dDt),
        
        float64(dRhoPhi),
    )
    