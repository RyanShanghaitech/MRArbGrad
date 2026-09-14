from numpy import *
from numpy.typing import *
from .main import solve

class TrajFunc:
    def __call__(self, p:float) -> NDArray:
        raise NotImplementedError("")
    
    def getPMin(self) -> float:
        raise NotImplementedError("")

    def getPMax(self) -> float:
        raise NotImplementedError("")

    def gradient(self) -> NDArray:
        return solve(self, self.getPMin(), self.getPMax())

class VDSpiral(TrajFunc):
    def __init__(self, denProf:NDArray, phi0:float=0.0):
        self.rho = asarray(denProf[0,:], dtype=float64)
        self.den = asarray(denProf[1,:], dtype=float64)
        self.phi0 = phi0

        drho = diff(self.rho)
        area = 0.5 * (self.den[:-1] + self.den[1:]) * drho
        self.phi = r_[0.0, cumsum(area)]

    def getPMin(self) -> float:
        return self.rho[0]

    def getPMax(self) -> float:
        return self.rho[-1]

    def evalPhi(self, rho:float) -> float:
        i = searchsorted(self.rho[:-1], rho, side="right") - 1

        den0, den1 = self.den[i], self.den[i+1]
        rho0, rho1 = self.rho[i], self.rho[i+1]
        slope = (den1 - den0) / (rho1 - rho0)
        drho = rho - rho0
        den = den0 + slope*drho;
    
        return self.phi[i] + (den0+den)*drho/2

    def __call__(self, rho:float) -> NDArray:
        angle = self.evalPhi(rho) + self.phi0

        return array([
            rho * cos(angle),
            rho * sin(angle),
        ])

class Spiral(VDSpiral):
    def __init__(self, den:float, phi0:float=0.0):
        super().__init__(array([[0.0, 0.5],[den, den]]), phi0)

class LVDSpiral(VDSpiral):
    def __init__(self, den0:float, den1:float, phi0:float=0.0):
        super().__init__(array([[0.0, 0.5],[den0, den1]]), phi0)

class Rosette(TrajFunc):
    def __init__(self, om1:float, om2:float, tmax:float, phi0:float=0.0):
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
    def __init__(
        self,
        den:float,
        tht0:float,
        phi0:float = 0.0,
    ):
        self.den = den
        self.tht0 = tht0
        self.phi0 = phi0

    def getPMin(self) -> float:
        return 0.0

    def getPMax(self) -> float:
        return self.den * 0.5

    def __call__(self, phi:float) -> NDArray:
        rho = phi / self.den
        angle = phi + self.phi0

        return rho * array([
            sin(self.tht0) * cos(angle),
            sin(self.tht0) * sin(angle),
            cos(self.tht0),
        ])

