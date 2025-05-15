from numpy import *
from matplotlib.pyplot import *
from scipy.special import ellipj, ellipk
from numpy.linalg import norm
import sympy as sp
import mrautograd as mag
from time import time

fov = 0.25
nPix = 256
dt = 10e-6
os = 10
sLim = 100*42.58e6*(fov/nPix)
gLim = 120e-3*42.58e6*(fov/nPix)
fExactSlew = True

# # Sp3d
# uPhi = nPix*5/16
# uTht = nPix*5/16
# symP = sp.Symbol("phi")
# expPhi = symP

# expTht = sp.sqrt(2*uPhi/uTht)*sp.sqrt(expPhi)
# expRho = sp.sqrt((uTht*uPhi)/(2*sp.pi**2*nPix**2))*sp.sqrt(expPhi)
# expX = expRho*sp.sin(expTht)*sp.cos(expPhi)
# expY = expRho*sp.sin(expTht)*sp.sin(expPhi)
# expZ = expRho*sp.cos(expTht)
# expK = sp.Array([expX,expY,expZ]).simplify()
# getK = sp.lambdify(symP, expK, "numpy")

# g0 = zeros((3,))
# p1 = (0.5**2)*2*(pi**2)*(nPix**2)/(uTht*uPhi)
# p0 = p1/nPix**2

# Yarnball
uPhi = nPix*5/16
uTht = nPix*5/16
symP = sp.Symbol("tht")
expTht = symP

expPhi = sp.sqrt(2*uTht/uPhi)*sp.sqrt(expTht)
expRho = uPhi/(2*sp.pi*nPix)*sp.sqrt(2*uTht/uPhi)*sp.sqrt(expTht)
expX = expRho*sp.sin(expTht)*sp.cos(expPhi)
expY = expRho*sp.sin(expTht)*sp.sin(expPhi)
expZ = expRho*sp.cos(expTht)

expK = sp.Array([expX,expY,expZ]).simplify()
getK = sp.lambdify(symP, expK, "numpy")
getK_D1p = None
getK_D2p = None

p1 = (pi*nPix)**2/(2*uPhi*uTht)
p0 = 10e-4 # we recommand not to start from 0 because the trajectory equation is undifferentrative at p=0.

# # Seiffert Spiral
# k = 0.07
# p1 = 20
# p0 = 0
# def getK(p:float64):
#     # sn, cn, _, _ = ellipj(p, k)
#     sn, cn = mag.calJacElip(p, k)
    
#     x = sn*cos(p*sqrt(k))
#     y = sn*sin(p*sqrt(k))
#     z = cn
    
#     return 0.5*p/(p1-p0)*array([x,y,z])

# # 3D Cones
# tht = pi/2 * 0.5
# dsamp = 64 # * tht/(pi/2)
# RhoOvPhi = 0.5/(2*pi) * dsamp/nPix
# p0 = 0 # phi
# p1 = 2*pi / (dsamp/nPix)
# def getK(p:float64):
#     phi = p
#     rho = RhoOvPhi * phi
#     x = rho * sin(tht) * cos(phi)
#     y = rho * sin(tht) * sin(phi)
#     z = rho * cos(tht)
#     return array([x,y,z])

# _ = p0
# p0 = p1
# p1 = _
print("getK(p0)", getK(p0))

# derive slew-rate constrained trajectory
t = time()
arrG = mag.calGrad(False, fov, nPix, sLim, gLim, dt, getK, getK_D1p, getK_D2p, p0, p1)
# arrG = mag.calGrad(p0, p1, getK, None, None, sLim, 1e8, dt)
t = time() - t
print(f"Exe Time: {t:.3f} s")
print(f"Wave Time: {arrG.shape[0]*dt*1e3:.3f} ms")
nRO, nAx = arrG.shape

arrS = diff(arrG, axis=0)/dt
print(f"sMax: {max(norm(arrS,axis=-1))/(42.58e6)*(nPix/fov)}")

arrDk = zeros((nRO+1,nAx))
arrDk[1:,:] = arrG*dt
arrK = getK(p0)*ones_like(arrDk)
arrK += cumsum(arrDk, axis=0)

# derive reference trajectory
print("p0", p0)
print("p1", p1)
arrP = arange(p0, p1, (p1-p0)*1e-4)
arrK_Ref = getK(arrP).T

figure(figsize=(18,9), dpi=120)

subplot(221, projection="3d")
plot(arrK[:,0], arrK[:,1], arrK[:,2], ".-")
plot(arrK_Ref[:,0], arrK_Ref[:,1], arrK_Ref[:,2], "-")
axis("equal")
grid("on")
title("k-Space")

subplot(223, projection="3d")
plot(arrG[:,0], arrG[:,1], arrG[:,2], ".-")
axis("equal")
grid("on")
title("g-Space")

subplot(222)
for iAx in range(3):
    plot(arrG[:,iAx]/(42.58e6)*(nPix/fov), ".-")
grid("on")
title("Gradient")

subplot(224)
plot(norm(arrS, axis=-1)/(42.58e6)*(nPix/fov), ".-")
ylim(sLim/(42.58e6)*(nPix/fov)*0.9, sLim/(42.58e6)*(nPix/fov)*1.1)
grid("on")
title(f"Slewrate, max:{max(norm(arrS,axis=-1))/(42.58e6)*(nPix/fov):.3f}")

figure(figsize=(9,9), dpi=120)
plot(norm(arrS, axis=-1)/(42.58e6)*(nPix/fov), ".-")
xlim(0,10)
ylim(sLim/(42.58e6)*(nPix/fov)*0.98, sLim/(42.58e6)*(nPix/fov)*1.02)
grid("on")
title("Slewrate")

show()
