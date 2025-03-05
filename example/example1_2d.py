from numpy import *
from numpy.linalg import norm
from matplotlib.pyplot import *
import sympy as sp
import g4n
from time import time

fov = 0.5
nPix = 256
dt = 10e-6
os = 10
sMax = 100*42.58e6*(fov/nPix)
fExactSlew = True

# # SpO
# nSp = 64
# p0 = 0
# p1 = (2*pi)*(nPix/2/nSp)
# symP = sp.Symbol("tht")
# A = (0.5/(2*pi))/(nPix/2/nSp)
# expRho = A*symP
# expX = expRho*sp.cos(symP)
# expY = expRho*sp.sin(symP)
# expK = sp.Array([expX,expY,1e-30*symP])

# getK = sp.lambdify(symP, expK, "numpy")
# getK_D1p = sp.lambdify(symP, expK.diff(symP,1).simplify(), "numpy")
# getK_D2p = sp.lambdify(symP, expK.diff(symP,2).simplify(), "numpy")

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

# Rosette
symP = sp.Symbol("t") 
om1 = 0.147*2*pi*2
om2 = 0.087*2*pi/2
p0 = 0
p1 = 15
expT = symP

expRho = 0.5*sp.sin(om1*expT)
expX = expRho*sp.cos(om2*expT)
expY = expRho*sp.sin(om2*expT)
expK = sp.Array([expX,expY,1e-30*symP])

getK = sp.lambdify(symP, expK, "numpy")
getK_D1p = sp.lambdify(symP, expK.diff(symP,1).simplify(), "numpy")
getK_D2p = sp.lambdify(symP, expK.diff(symP,2).simplify(), "numpy")

# derive slew-rate constrained trajectory
t = time()
g4n.init(sMax, 1e8, dt, os, 0, 0, False)
arrG = g4n.compute(getK, p0, p1)
t = time() - t
print(f"Exe Time: {t}")
nRO, nAx = arrG.shape

arrS = diff(arrG, axis=0)/dt
print(f"sMax: {max(norm(arrS,axis=-1))/(42.58e6)*(nPix/fov)}")

arrDk = zeros((nRO+1,nAx))
arrDk[1:,:] = arrG*dt
arrK = getK(p0)*ones_like(arrDk)
arrK += cumsum(arrDk, axis=0)

# derive reference trajectory
arrTht = linspace(p0, p1, int(1e6))
arrK_Ref = getK(arrTht).T
print(arrK_Ref.shape)

# plot
figure(figsize=(12,6), dpi=120)

subplot(221)
plot(arrK[:,0], arrK[:,1], ".-", label="K_Imp")
plot(arrK_Ref[:,0], arrK_Ref[:,1], "--", label="K_Ref")
axis("equal")
xlim(-0.001,0.001)
ylim(-0.001,0.001)
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
ylim(sMax/(42.58e6)*(nPix/fov)*0.9, sMax/(42.58e6)*(nPix/fov)*1.1)
grid("on")
title(f"Slewrate, max:{max(norm(arrS,axis=-1))/(42.58e6)*(nPix/fov):.3f}")

show()
