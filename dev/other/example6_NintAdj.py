from numpy import *
from matplotlib.pyplot import *
import mrarbgrad as mag
from numpy.linalg import norm
from numpy.random import uniform

# adjust number of interleaves, may depends on Nyquist interval
nInt = 100

fov = 0.22
nPix = 258
sLim = 100 * 42.5756e6 * fov / nPix
gLim = 120e-3 * 42.5756e6 * fov / nPix
dt = 10e-6
m = 0.07 # measuared by discrepancy test
uMax = 20 # meansured by readout duration
dtGrad = 10e-6
dtADC = 2.5e-6

def getK(u:float64) -> ndarray:
    sn, cn = mag._calJacElip(u, m)
    phi = sqrt(m)*u
    rho = 0.5*((u/uMax)**1)
    return rho*array([sn*cos(phi), sn*sin(phi), cn], dtype=float64).T
arrG = mag.calGrad(False, fov, nPix, sLim, gLim, dt, getK, None, None, 0, uMax)
print(f"readout: {arrG.shape[0]*dtGrad*1000} ms")
arrK = cumsum(arrG*dtGrad, axis=0)

# ensures trajectory ends at (0, 0, 0.5)
arrG = mag.rotate(arrG, -arctan2(arrK[-1,1],arrK[-1,0]), 2)
arrK = cumsum(arrG*dtGrad, axis=0)
arrG = mag.rotate(arrG, -arctan2(arrK[-1,0],arrK[-1,2]), 1)
arrK = cumsum(arrG*dtGrad, axis=0)
arrG_Ref = arrG.copy()
arrK_Ref = arrK.copy()

# generate Fibonacci points
arrFib = mag._calSphFibPt(nInt)

# generate other interleaves
lstArrG = []
lstArrK = []
for iInt in range(nInt):
    if iInt%1000==999: print(f"{iInt+1}/{nInt}")
    arrG = arrG_Ref.copy()
    arrG = mag.rotate(arrG, uniform(-pi, pi), 2) # randomly rotate along z-axis
    arrG = mag.rotate(arrG, arctan2(norm(arrFib[iInt,:2]),arrFib[iInt,2]), 1) # apply theta
    arrG = mag.rotate(arrG, arctan2(arrFib[iInt,1],arrFib[iInt,0]), 2) # apply phi
    arrK = cumsum(arrG*dtGrad, axis=0)
    lstArrG.append(arrG)
    lstArrK.append(arrK)

# plot
fig = figure()
ax = fig.add_subplot(111)
arrG = lstArrG[0]
ax.plot(norm(arrG[1:,:]-arrG[:-1,:], axis=1)/dtGrad/42.58e6*nPix/fov, ".-")
ax.set_ylim(90,110)
ax.grid("on")

fig = figure(figsize=(6,6), dpi=120)
ax = fig.add_subplot(111, projection="3d")
# lstArrK = [lstArrK[0], lstArrK[-1]]
for arrK in lstArrK:
    # ax.clear()
    # ax.plot(arrK[:,0], arrK[:,1], arrK[:,2], ".-", markersize=2, linewidth=1)
    ax.plot(arrK[-1:,0], arrK[-1:,1], arrK[-1:,2], ".-")
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.5,0.5)
    ax.axis("equal")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_zlabel("kz")
    # show(block=False)
    # pause(1e-1)

show()
