from numpy import *
from numpy.typing import *
from typing import *
from .main import solve

class TrajFunc:
    """
    Template class for trajectory functions. Functions need to be implemented are:
    - `__call__()`: the function that defines the shape of a trajectory.
    - `getPMin()` returns the lower bound of the trajectory parameter.
    - `getPMax()` returns the upper bound of the trajectory parameter

    Call `.gradient()` to get the corresponding gradient waveform.
    """
    def __call__(self, p:float) -> NDArray:
        raise NotImplementedError("")
    
    def getPMin(self) -> float:
        raise NotImplementedError("")

    def getPMax(self) -> float:
        raise NotImplementedError("")

    def gradient(self) -> NDArray:
        return solve(self, self.getPMin(), self.getPMax())

class VDSpiral(TrajFunc):
    """
    Variable Density Spiral trajectory.
    Call `.gradient()` to get the corresponding gradient waveform.

    Args:
        denProf (NDArray): density profile, `denProf[0,:]` defines the radius, `denProf[1,:]` defines the derivative of the polar angle w.r.t. the radius.
        phi0 (float): initial polar angle.
    """
    def __init__(self, denProf:NDArray, phi0:float=0.0):
        self.rho = asarray(denProf[0,:], dtype=float64)
        self.den = asarray(denProf[1,:], dtype=float64)
        self.phi0 = phi0

        drho = diff(self.rho)
        area = (self.den[:-1] + self.den[1:]) * drho / 2
        self.phi = hstack(([0.0], cumsum(area)))

    def getPMin(self) -> float:
        return self.rho[0]

    def getPMax(self) -> float:
        return self.rho[-1]

    def _evalPhi(self, rho:float) -> float:
        i = searchsorted(self.rho[:-1], rho, side="right") - 1

        den0, den1 = self.den[i], self.den[i+1]
        rho0, rho1 = self.rho[i], self.rho[i+1]
        slope = (den1 - den0) / (rho1 - rho0)
        drho = rho - rho0
        den = den0 + slope*drho;
    
        return self.phi[i] + (den0+den)*drho/2

    def __call__(self, rho:float) -> NDArray:
        phi = self._evalPhi(rho) + self.phi0
        return rho*array((cos(phi),sin(phi)))

class Spiral(VDSpiral):
    """
    Spiral trajectory.
    Call `.gradient()` to get the corresponding gradient waveform.

    Args:
        den (float): density, refers to the derivative of polar angle (phi) w.r.t. radius (rho).
        phi0 (float): initial polar angle.
    """
    def __init__(self, den:float, phi0:float=0.0):
        super().__init__(array([[0.0, 0.5],[den, den]]), phi0)

class DDSpiral(VDSpiral):
    """
    Dual Density Spiral trajectory.
    Call `.gradient()` to get the corresponding gradient waveform.

    Args:
        den0 (float): derivative of polar angle (phi) w.r.t. radius (rho) at the first radius.
        den1 (float): derivative of polar angle (phi) w.r.t. radius (rho) at the second radius.
        rho0 (float): radius (rho) at the first radius.
        rho1 (float): radius (rho) at the second radius.
        phi0 (float): initial polar angle.
        decay (Literal): how does `den0` decay to `den1`.
        nSamp (int): how many samples are used to form the density profile.
    """
    def __init__(self, den0:float, den1:float, rho0:float=0.0, rho1:float=0.5, phi0:float=0.0, decay:Literal["exp", "iprop", "prop"]="iprop", nSamp:int=10000):
        rho = linspace(rho0, rho1, nSamp)
        den = linspace(0, 1, nSamp)
        if decay=="exp": den = den0 * (den1/den0)**den
        elif decay=="iprop": den = 1/linspace(1/den0,1/den1,nSamp)
        elif decay=="prop": den = linspace(den0,den1,nSamp)
        else: raise ValueError("decay")

        if rho[0]==0: rho=rho[1:]; den=den[1:]
        if rho[-1]==0.5: rho=rho[:-1]; den=den[:-1]

        rho = hstack((0, rho, 0.5))
        den = hstack((den0, den, den1))
        super().__init__(vstack((rho,den)), phi0)
        self.rho, self.den = rho, den # test

class Rosette(TrajFunc):
    """
    Rosette trajectory.
    Call `.gradient()` to get the corresponding gradient waveform.

    Args:
        om1 (float): angular frequency of the change of radius.
        om2 (float): angular frequency of the change of polar angles.
        tmax (float): upper bound of the trajectory parameter.
        phi0 (float): initial polar angle.
    """
    def __init__(self, om1:float, om2:float, tmax:float=1.0, phi0:float=0.0):
        self.om1 = om1
        self.om2 = om2
        self.tmax = tmax
        self.phi0 = phi0

    def getPMin(self) -> float:
        return 0.0

    def getPMax(self) -> float:
        return self.tmax

    def __call__(self, t:float) -> NDArray:
        rho = 0.5 * sin(self.om1 * t)
        phi = self.om2 * t + self.phi0

        return array([
            rho * cos(phi),
            rho * sin(phi),
        ])

class Yarnball(TrajFunc):
    """
    Yarnball trajectory.
    Call `.gradient()` to get the corresponding gradient waveform.

    Args:
        den (float): derivative of polar angle (phi) w.r.t. radius (rho)
        tht0 (float): initial azimuth angle.
        phi0 (float): initial polar angle.
    """
    def __init__(self, den:float, tht0:float, phi0:float):
        self.den = den
        self.tht0 = tht0
        self.phi0 = phi0

    def __call__(self, thtSqrt:float) -> NDArray:
        tht = sign(thtSqrt) * thtSqrt**2
        rho = sqrt(2.0) * thtSqrt / self.den
        phi = sqrt(2.0) * thtSqrt

        tht += self.tht0
        phi += self.phi0

        return rho * array([
            sin(tht) * cos(phi),
            sin(tht) * sin(phi),
            cos(tht),
        ])

    def getPMin(self) -> float:
        return 0.0

    def getPMax(self) -> float:
        return self.den / sqrt(8)


class Cones(TrajFunc):
    """
    Cones trajectory.
    Call `.gradient()` to get the corresponding gradient waveform.

    Args:
        den (float): derivative of polar angle (phi) w.r.t. radius (rho)
        tht0 (float): initial azimuth angle.
        phi0 (float): initial polar angle.
    """
    def __init__(self, den:float, tht0:float, phi0:float=0.0):
        self.den = den
        self.tht0 = tht0
        self.phi0 = phi0

    def getPMin(self) -> float:
        return 0.0

    def getPMax(self) -> float:
        return self.den * 0.5

    def __call__(self, phi:float) -> NDArray:
        rho = phi / self.den
        phi = phi + self.phi0

        return rho * array([
            sin(self.tht0) * cos(phi),
            sin(self.tht0) * sin(phi),
            cos(self.tht0),
        ])

