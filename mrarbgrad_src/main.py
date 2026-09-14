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

def solve(*args, **kwargs) -> NDArray|None:
    if isinstance(args[0], ndarray):
        return ext.solve_samp(args[0])
    elif callable(args[0]):
        return ext.solve_func(args[0], float(args[1]), float(args[2]))
    else:
        raise TypeError(f"type(args[0]): {type(args[0])}")

def scan(traj:str, nPix:int, nAcq:int) -> List[Tuple[float, NDArray, float]]:
    """
    run built-in scan plan.

    Args:
        traj (str): plan name
        nPix (int): number of pixels per dimension
        nAcq (int): number of acquisition

    Returns:
        List[Tuple[float, NDArray, float]]: list of (k0, grad, k1) tuple; k0: phase encoding moment; grad: readout gradient waveform; k1: final moment.
    """
    return ext.scan(str(traj), int(nPix), int(nAcq))

def config(dt:float=10e-6, ovsp:int=4, sLim:float=tm2hzpx(50,1e-3), gLim:float=tm2hzpx(30e-3,1e-3), g0Norm:float=0.0, g1Norm:float=0.0, enTrajRep:bool=True, enGradRep:bool=True, lenGradRsv:int=int(1e5), lenTrajRsv:int=int(1e4)):
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
        lenGradRsv (int): reserved buffer size for gradient computation
        lenTrajRsv (int): reserved buffer size for trajectory reparameterization
    """
    ext.config(float(dt), int(ovsp), float(sLim), float(gLim), float(g0Norm), float(g1Norm), bool(enTrajRep), bool(enGradRep), int(lenGradRsv), int(lenTrajRsv))

def saveF64(hdr:str, bin:str, arr:NDArray) -> bool:
    """
    save vector file (float64)

    Args:
        hdr (str): header (hdr) file path
        bin (str): bin file path
        arr (NDarray): array to be saved

    Returns:
        bool: True for success
    """
    return ext.saveF64(str(hdr), str(bin), arr)

def loadF64(hdr:str, bin:str) -> list[NDArray]|None:
    """
    load vector file (float64)

    Args:
        hdr (str): header (hdr) file path
        bin (str): bin file path

    Returns:
        list[NDArray]: list of loaded array
    """
    return ext.loadF64(str(hdr), str(bin))

def saveF32(hdr:str, bin:str, arr:NDArray) -> bool:
    """
    save vector file (float32)

    Args:
        hdr (str): header (hdr) file path
        bin (str): bin file path
        arr (NDarray): array to be saved

    Returns:
        bool: True for success
    """
    return ext.saveF32(str(hdr), str(bin), arr)

def loadF32(hdr:str, bin:str) -> list[NDArray]|None:
    """
    load vector file (float32)

    Args:
        hdr (str): header (hdr) file path
        bin (str): bin file path

    Returns:
        list[NDArray]: list of loaded array
    """
    return ext.loadF32(str(hdr), str(bin))
