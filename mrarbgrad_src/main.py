from numpy import *
from numpy.typing import *
from typing import *
from builtins import bool
import mrarbgrad.ext as ext
from .utility import tm2hzpx

@overload
def solve(func:Callable, p0:float, p1:float) -> NDArray:
    """
    Solve the gradient waveform for a trajectory function.

    Args:
        func (Callable): trajectory function
        p0 (float): parameter lower bound
        p1 (float): parameter higher bound

    Returns:
        NDArrray: gradient waveform
    """
    ...

@overload
def solve(samp:NDArray) -> NDArray:
    """
    Solve the gradient waveform for a trajectory point set.

    Args:
        samp (NDArray): trajectory point set

    Returns:
        NDArrray: gradient waveform
    """
    ...

def solve(traj:Callable|NDArray, p0:float|None=None, p1:float|None=None) -> NDArray|None:
    if callable(traj): return ext.solve_func(traj, float(p0), float(p1))
    elif isinstance(traj, ndarray): return ext.solve_samp(traj)
    else: raise TypeError(f"type(traj): {type(traj)}")

def scan(traj:Literal["Spiral", "DDSpiral", "Rosette", "RosetteClassic", "Yarnball", "Cones"], nPix:int, nAcq:int) -> List[Tuple[NDArray, NDArray, NDArray]]:
    """
    run built-in scan plan.

    Args:
        traj (str): plan name
        nPix (int): number of pixels per dimension
        nAcq (int): number of acquisition

    Returns:
        List[Tuple[NDArray, NDArray, NDArray]]: list of (k0, grad, k1) tuple; k0: phase encoding moment; grad: readout gradient waveform; k1: final moment.
    """
    return ext.scan(str(traj), int(nPix), int(nAcq))

def config(dt:float=10e-6, ovsp:int=4, sLim:float=tm2hzpx(50,1e-3), gLim:float=tm2hzpx(30e-3,1e-3), g0Norm:float=0.0, g1Norm:float=0.0, enTrajRep:bool=True, enGradRep:bool=True, lenTrajRsv:int=int(1e4), lenGradRsv:int=int(1e5)):
    """
    set solver configurations.

    Args:
        dt (float): gradeint time resolution in `s`
        ovsp (int): oversampling ratio
        sLim (float): slew-rate limit amplitude in `Hz/px/s`
        gLim (float): gradeint amplitude limit in `Hz/px`
        g0Norm (float): desired initial gradient amplitude
        g1Norm (float): desired final gradient amplitude
        enTrajRep (bool): enable trajectory reparameterization
        enGradRep (bool): enable gradient reparameterization
        lenTrajRsv (int): reserved buffer size for trajectory reparameterization
        lenGradRsv (int): reserved buffer size for gradient computation
    """
    ext.config(float(dt), int(ovsp), float(sLim), float(gLim), float(g0Norm), float(g1Norm), bool(enTrajRep), bool(enGradRep), int(lenTrajRsv), int(lenGradRsv))

def saveF64(hdr:str, bin:str, arrs:List[NDArray]):
    """
    save as a vector file (float64)

    Args:
        hdr (str): header (hdr) file path
        bin (str): bin file path
        arrs (List[NDarray]): arrays to be saved
    """
    ext.saveF64(str(hdr), str(bin), arrs)

def loadF64(hdr:str, bin:str) -> list[NDArray]:
    """
    load a vector file (float64)

    Args:
        hdr (str): header (hdr) file path
        bin (str): bin file path

    Returns:
        list[NDArray]: list of loaded array
    """
    return ext.loadF64(str(hdr), str(bin))

def saveF32(hdr:str, bin:str, arrs:List[NDArray]):
    """
    save as a vector file (float32)

    Args:
        hdr (str): header (hdr) file path
        bin (str): bin file path
        arrs (List[NDArray]): arrays to be saved
    """
    ext.saveF32(str(hdr), str(bin), arrs)

def loadF32(hdr:str, bin:str) -> list[NDArray]:
    """
    load a vector file (float32)

    Args:
        hdr (str): header (hdr) file path
        bin (str): bin file path

    Returns:
        list[NDArray]: list of loaded array
    """
    return ext.loadF32(str(hdr), str(bin))
