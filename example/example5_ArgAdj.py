from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import g4n

# adjust argument range of Seiffert Spiral, depends on desired acq. time
uMax = 30

fov = 0.5
nPix = 256
m = 0.07

def getK(u:float64) -> ndarray:
    sn, cn = g4n.calJacElip(u, m)
    phi = sqrt(m)*u
    rho = 0.5*((u/uMax)**1)
    return rho*array([sn*cos(phi), sn*sin(phi), cn], dtype=float64).T
g4n.init(100*42.58e6*(fov/nPix), inf, 10e-6)
arrG = g4n.compute(getK, 0, uMax)
arrK = g4n.cvtGrad2Traj(arrG, 10e-6, 2.5e-6)
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