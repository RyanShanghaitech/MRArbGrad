import g4n
from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
from time import time

fov = 0.20
nPix = 256
dt = 10e-6
sLim = 100 * 42.5756e6 * fov/nPix
gLim = 120e-3 * 42.5756e6 * fov/nPix

t = time()
lstArrG = [*g4n.Function.getG_Spiral(sLim, gLim, kRhoPhi=0.5/(4*pi))] # 100.011
# lstArrG = [*g4n.Function.getG_VarDenSpiral(sLim, gLim)] # 100.010
# lstArrG = [*g4n.Function.getG_Rosette(sLim/4, gLim)]; sLim /= 4 # 25.003
# lstArrG = [*g4n.Function.getG_CloseSpiral(sLim, gLim)] # 99.998
# lstArrG = [*g4n.Function.getG_Spiral3d(sLim, gLim, kRhoTht=0.5/(2*pi))] # 99.990
# lstArrG = [*g4n.Function.getG_Yarnball(sLim, gLim, kRhoPhi=0.5/(2*pi))] # 100.012
# lstArrG = [*g4n.Function.getG_Seiffert(sLim, gLim)] # 100.012
# lstArrG = [*g4n.Function.getG_Cones(sLim, gLim, tht0=100*(1/256)/0.5)] # 100.006
t = time() - t
print(f"Exe Time: {t}")
print(f"IntLea Num.: {len(lstArrG)}")

arrG = lstArrG[0]
nRO, nAx = arrG.shape

arrS = diff(arrG, axis=0)/dt
print(f"sMax: {max(norm(arrS,axis=-1))/(42.58e6)*(nPix/fov)}")

lstArrK = []
for arrG in lstArrG:
    arrDk = zeros((nRO+1,nAx))
    arrDk[1:,:] = arrG*dt
    arrK = zeros_like(arrDk)
    arrK += cumsum(arrDk, axis=0)
    lstArrK.append(arrK[:])
lstArrK = lstArrK[:1]

# plot
figure(figsize=(18,9), dpi=120)

subplot(261)
for arrK in lstArrK: plot(*arrK.T[(0,1),:], ".-")
axis("equal")
grid("on")
title(f"kx-ky {1}/{len(lstArrG)}")

subplot(262)
for arrK in lstArrK: plot(*arrK.T[(0,2),:], ".-")
axis("equal")
grid("on")
title("kx-kz")

subplot(263)
for arrK in lstArrK: plot(*arrK.T[(1,2),:], ".-")
axis("equal")
grid("on")
title("ky-kz")

subplot(222)
for iAx in range(nAx):
    plot(arrG[:,iAx]/(42.58e6)*(nPix/fov), ".-")
grid("on")
title("Gradient")

subplot(267)
plot(*arrG.T[(0,1),:], ".-")
axis("equal")
grid("on")
title("gx-gy")

subplot(268)
plot(*arrG.T[(0,2),:], ".-")
axis("equal")
grid("on")
title("gx-gz")

subplot(269)
plot(*arrG.T[(1,2),:], ".-")
axis("equal")
grid("on")
title("gy-gz")

subplot(224)
plot(norm(arrS,axis=-1)/(42.58e6)*(nPix/fov), ".-")
ylim(sLim/(42.58e6)*(nPix/fov)*0.9, sLim/(42.58e6)*(nPix/fov)*1.1)
grid("on")
title(f"Slewrate, max:{max(norm(arrS,axis=-1))/(42.58e6)*(nPix/fov):.3f}")

subplots_adjust(0.05,0.1,0.95,0.9, 0.2, 0.2)

show()
