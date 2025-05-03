from numpy import *
from numpy.linalg import norm
from matplotlib.pyplot import *
import sympy as sp
import mrautograd as mag
from time import time

fov = 0.256
nPix = 256
dt = 10e-6
os = 10
sLim = 100 * 42.5756e6 * fov / nPix
gLim = 120e-3 * 42.5756e6 * fov / nPix

# SpO
kRhoPhi = 0.5/(8*pi)
p0 = 0
p1 = 0.5/kRhoPhi
symP = sp.Symbol("tht")
expRho = kRhoPhi*symP
expX = expRho*sp.cos(symP)
expY = expRho*sp.sin(symP)
expK = sp.Array([expX,expY,1e-30*symP])

getK = sp.lambdify(symP, expK, "numpy")
getK_D1p = sp.lambdify(symP, expK.diff(symP,1).simplify(), "numpy")
getK_D2p = sp.lambdify(symP, expK.diff(symP,2).simplify(), "numpy")

# # SpI
# nSp = 64
# p0 = (2*pi)*(nPix/2/nSp)
# p1 = p0*1e-4 # -(2*pi)*(nPix/2/nSp)
# symP = sp.Symbol("tht")
# A = (0.5/(2*pi))/(nPix/2/nSp)
# expTht = symP
# expRho = A*expTht
# expX = expRho*sp.cos(expTht)
# expY = expRho*sp.sin(expTht)
# expK = sp.Array([expX,expY,1e-30*symP])

# getK = sp.lambdify(symP, expK, "numpy")
# getK_D1p = sp.lambdify(symP, expK.diff(symP,1).simplify(), "numpy")
# getK_D2p = sp.lambdify(symP, expK.diff(symP,2).simplify(), "numpy")

# # Rosette
# symP = sp.Symbol("t") 
# om1 = 9*pi
# om2 = 19*pi
# p0 = 0
# p1 = 1
# expT = symP

# expRho = 0.5*sp.sin(om1*expT)
# expX = expRho*sp.cos(om2*expT)
# expY = expRho*sp.sin(om2*expT)
# expK = sp.Array([expX,expY,1e-30*symP])

# getK = sp.lambdify(symP, expK, "numpy")
# getK_D1p = sp.lambdify(symP, expK.diff(symP,1).simplify(), "numpy")
# getK_D2p = sp.lambdify(symP, expK.diff(symP,2).simplify(), "numpy")

# # VDS
# nSp0 = 32
# nSp1 = 8
# A0 = (0.5/(2*pi))/(nPix/2/nSp0)
# A1 = (0.5/(2*pi))/(nPix/2/nSp1)
# print("A0", A0)
# print("A1", A1)
# def getK(tht:float64):
#     rho = A0*(exp(2*(A1 - A0)*tht) - 1) / (2*(A1 - A0))
#     return array([rho*cos(tht), rho*sin(tht), 1e-30*tht])
# p0 = 0
# p1 = (log(A1)-log(A0)) / (2*(A1-A0))

# derive slew-rate constrained trajectory
t = time()
arrG = g4n.calGrad(False, fov, nPix, sLim, gLim, dt, getK, None, None, p0, p1)
t = time() - t
print(f"Exe Time: {t}")
nRO, nAx = arrG.shape

arrS = diff(arrG, axis=0)/dt
print(f"sMax: {max(norm(arrS,axis=-1))/(42.58e6)*(nPix/fov)}")

arrDk = zeros((nRO+1,nAx))
arrDk[1:,:] = arrG*dt
arrK = getK(p0)*ones_like(arrDk)
arrK += cumsum(arrDk, axis=0)
# arrK = g4n.cvtGrad2Traj(arrG, dt, dt)

# derive reference trajectory
arrTht = linspace(p0, p1, int(1e6))
arrK_Ref = getK(arrTht).T

# plot
figure(figsize=(20,10), dpi=120)

subplot(221)
plot(arrK[:,0], arrK[:,1], ".-", label="K_Imp")
plot(arrK_Ref[:,0], arrK_Ref[:,1], "--", label="K_Ref")
xlim(-0.5,0.5)
ylim(-0.5,0.5)
axis("equal")
grid("on")
legend()
title("k-Space")

subplot(222)
for iAx in range(2):
    plot(arrG[:,iAx]/(42.58e6)*(nPix/fov), ".-")
grid("on")
title("Gradient")

subplot(223)
plot(arrG[:,0], arrG[:,1], ".-")
axis("equal")
grid("on")
title("g-Space")

subplot(224)
plot(norm(arrS,axis=-1)/(42.58e6)*(nPix/fov), ".-")
ylim(sLim/(42.58e6)*(nPix/fov)*0.9, sLim/(42.58e6)*(nPix/fov)*1.1)
grid("on")
title(f"Slewrate, max:{max(norm(arrS,axis=-1))/(42.58e6)*(nPix/fov):.3f}")

show()
