from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import mrautograd as mag

# adjust argument range of Seiffert Spiral, depends on desired acq. time
uMax = 30

fov = 0.5
nPix = 256
sLim = 100 * 42.5756e6 * fov / nPix
gLim = 120e-3 * 42.5756e6 * fov / nPix
dt = 10e-6
m = 0.07

def getK(u:float64) -> ndarray:
    sn, cn = mag.calJacElip(u, m)
    phi = sqrt(m)*u
    rho = 0.5*((u/uMax)**1)
    return rho*array([sn*cos(phi), sn*sin(phi), cn], dtype=float64).T

arrG = mag.calGrad(False, fov, nPix, sLim, gLim, dt, getK, None, None, 0, uMax)
arrK = mag.cvtGrad2Traj(arrG, 10e-6, 2.5e-6)
tRO = arrG.shape[0]*10e-6

# plot
fig = figure()

ax = fig.add_subplot(111, projection="3d")
ax.clear()
ax.plot(arrK[:,0], arrK[:,1], arrK[:,2], ".-")#, markersize=0.5, linewidth=1)
ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.5,0.5)
ax.set_zlim(-0.5,0.5)
ax.axis("equal")
ax.set_title(f"tRO: {tRO*1e3:.3f} ms")

show()