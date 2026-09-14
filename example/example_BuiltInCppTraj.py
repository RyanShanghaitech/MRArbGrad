from numpy import *
from numpy.typing import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import mrarbgrad as mag

# parameters
fov, nPix, nAx = 0.320, 320, 2
dtGrad = 10e-6
dtAdc = 2.5e-6
nAcq = 1000
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

# pull a scan plan from library
lstK0GradK1 = mag.scan("RosetteClassic", nPix, nAcq)
print(f"nAcq: {len(lstK0GradK1)}")
lstArrGrad, lstArrK = [], []
for k0, arrGrad, k1 in lstK0GradK1:
    arrK = mag.integrate(arrGrad, dtGrad, dtAdc)
    arrK += k0
    lstArrGrad.append(arrGrad)
    lstArrK.append(arrK)

# visualize
iAcq = 300
arrGrad:NDArray = mag.hzpx2tm(lstArrGrad[iAcq], fov/nPix)*1e3
arrK:NDArray = lstArrK[iAcq]

figure(figsize=(3,3), dpi=300)
plot(arrK[:,0], arrK[:,1], ".-")
xlim(-0.5,0.5)
ylim(-0.5,0.5)
axis("equal")
grid(True)
title("Trajectory")

figure(figsize=(6,3), dpi=300)
for iAx in range(nAx):
    plot(arrGrad[:,iAx], ".-")
ylabel("mT/m")
grid(True)
title("Gradient")

show()
