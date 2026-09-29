from numpy import *
from numpy.typing import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import mrarbgrad as mag

# parameters
fov, nPix = 0.320, 320
dtGrad = 10e-6
dtAdc = 2.5e-6
mag.config(
    dt=10e-6,
    ovsp=4,
    sLim=mag.tm2hzpx(50,1e-3),
    gLim=mag.tm2hzpx(30e-3,1e-3),
    g0Norm=0.0,
    g1Norm=0.0,
    enTrajRep=True,
    enGradRep=True,
    lenGradRsv=1e5,
    lenTrajRsv=1e4
)

# define a customized trajectory
om1, om2, phi0 = 5*pi, 3*pi, 0.0
tMin, tMax = 0.0, 1.0
def Rosette(t:float) -> NDArray:
    rho = 0.5 * sin(om1*t)
    phi = om2*t + phi0
    return array([rho*cos(phi), rho*sin(phi)], dtype=float64)
arrGrad = mag.solve(Rosette, tMin, tMax)
arrK = mag.integrate(arrGrad, dtGrad, dtAdc)

# visualize
arrGrad:NDArray = mag.hzpx2tm(arrGrad, fov/nPix)*1e3

figure(figsize=(3,3), dpi=300)
plot(arrK[:,0], arrK[:,1], ".-")
xlim(-0.5,0.5)
ylim(-0.5,0.5)
axis("equal")
grid(True)
title("Trajectory")
savefig(__file__.replace(".py", "_fig1.png"))

figure(figsize=(6,3), dpi=300)
for iAx in range(arrGrad.shape[-1]):
    plot(arrGrad[:,iAx], ".-")
ylabel("mT/m")
grid(True)
title("Gradient")
savefig(__file__.replace(".py", "_fig2.png"))

# show()
