from numpy import *
from numpy.typing import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import mrarbgrad as mag

# parameters
fov, nPix, nAx = 0.320, 320, 2
dtGrad = 10e-6
dtAdc = 2.5e-6
sLim = mag.tm2hzpx(50,fov/nPix)
gLim = mag.tm2hzpx(30e-3,fov/nPix)
mag.config(
    dt=dtGrad,
    ovsp=4,
    sLim=sLim,
    gLim=gLim,
    g0Norm=0.0,
    g1Norm=0.0,
    enTrajRep=False, 
    enGradRep=True,
    lenTrajRsv=1e4,
    lenGradRsv=1e5
)

# pull a trajectory from library
traj = mag.DDSpiral(32*pi/0.5, 2*pi/0.5, 0.1, 0.5, decay="iprop")

rho, den = traj.rho, traj.den
figure(dpi=300)
plot(rho, den, ".-")
savefig(__file__.replace(".py", "_fig0.png"))

arrGrad = traj.gradient()
arrGrad = arrGrad[:,:nAx]
arrK = mag.integrate(arrGrad, dtGrad, dtAdc)
arrSlew = diff(arrGrad, axis=0)/dtGrad

maxGrad = norm(arrGrad, axis=-1).max()
maxSlew = norm(arrSlew, axis=-1).max()
print(f"grad overshot: {(maxGrad-gLim)/gLim*100:.3f}%")
print(f"slew overshot: {(maxSlew-sLim)/sLim*100:.3f}%")

# visualize
arrGrad:NDArray = mag.hzpx2tm(arrGrad, fov/nPix)*1e3
arrSlew:NDArray = mag.hzpx2tm(arrSlew, fov/nPix)

figure(figsize=(3,3), dpi=300)
plot(arrK[:,0], arrK[:,1], ".-")
xlim(-0.5,0.5)
ylim(-0.5,0.5)
axis("equal")
grid(True)
title("Trajectory")
savefig(__file__.replace(".py","_fig1.png"))

figure(figsize=(6,3), dpi=300)
for iAx in range(nAx):
    plot(arrGrad[:,iAx], ".-")
ylabel("mT/m")
grid(True)
title("Gradient")
savefig(__file__.replace(".py","_fig2.png"))

figure(figsize=(6,6), dpi=300)
ax = subplot(211)
ax.plot(norm(arrGrad,axis=-1), ".-")
ax.set_ylabel("mT/m")
ax = subplot(212)
ax.plot(norm(arrSlew,axis=-1), ".-")
ax.set_ylim(mag.hzpx2tm(sLim*0.9, fov/nPix), mag.hzpx2tm(sLim*1.1, fov/nPix))
ax.set_ylabel("T/m/s")
savefig(__file__.replace(".py","_fig3.png"))

# show()
