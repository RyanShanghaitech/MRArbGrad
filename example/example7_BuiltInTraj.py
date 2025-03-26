import g4n
from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
from time import time
import finufft as fn
import fars

fov = 0.20
nPix = 256
dtGrad = 10e-6
dtADC = 5e-6
sLim = 100 * 42.5756e6 * fov/nPix
gLim = 120e-3 * 42.5756e6 * fov/nPix

# calculate gradient
t = time()

lstArrGrad = g4n.Function.getG_Spiral(sLim, gLim) # 0.380s

# lstArrGrad = g4n.Function.getG_VarDenSpiral(sLim, gLim) # 0.499s

# lstArrGrad = g4n.Function.getG_Rosette(sLim/4, gLim); sLim /= 4 # 16.39s (9 frames)

# lstArrGrad = g4n.Function.getG_CloseSpiral(sLim, gLim) # 0.462s

# lstArrGrad = g4n.Function.getG_Shell3d(sLim, gLim) # 183.7

# lstArrGrad = g4n.Function.getG_Yarnball(sLim, gLim) # 196.1
    
# lstArrGrad = g4n.Function.getG_Seiffert(sLim, gLim) # 232.9s

# lstArrGrad = g4n.Function.getG_Cones(sLim, gLim) # 149.9s

t = time() - t
print(f"Exe Time: {t}")
print(f"Intlea Num.: {len(lstArrGrad)}")

nRO_Max = max(arrG.shape[0] for arrG in lstArrGrad)
tTR = (nRO_Max*dtGrad + 5e-3)
tScan = tTR*len(lstArrGrad)
print(f"Tscan {tScan:.3e} s")

# derive shape parameter
if all(lstArrGrad[0][:,2]==0): lstArrGrad = [arrG[:,:2] for arrG in lstArrGrad]
nRO, nAx = lstArrGrad[0].shape

# derive slewrate
arrSlew = diff(lstArrGrad[0], axis=0)/dtGrad
print(f"sMax: {max(norm(arrSlew,axis=-1))/(42.58e6)*(nPix/fov)}")

# derive trajectory
lstArrK = []
for arrGrad in lstArrGrad:
    arrK = g4n.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
    arrRho = norm(arrK, axis=-1)
    lstArrK.append(arrK)

# simulate phantom
arrI = asarray(load("./resource/arrM0.npz")["arrM0"])
if nAx == 2: arrI = arrI[nPix//2,:,:]
arrX = array(meshgrid\
    (
        arange(-nPix//2, nPix//2, 1),
        arange(-nPix//2, nPix//2, 1),
        indexing="ij"
    )).T.reshape(-1,2)
arrK = concatenate(lstArrK, axis=0)
arrDcf = fars.calDcf(nPix, arrK).astype(complex64)
arrOm = 2*pi*arrK; arrOm = arrOm.astype(float32)

plan = fn.Plan(2, tuple(nPix for _ in range(nAx)), isign=-1, dtype="complex64")
plan.setpts(*arrOm.T)
arrS = plan.execute(arrI.astype(complex64))

plan = fn.Plan(1, tuple(nPix for _ in range(nAx)), isign=1, dtype="complex64")
plan.setpts(*arrOm.T)
arrI_Reco = plan.execute(arrS*arrDcf)

if nAx==2:
    figure()
    
    subplot(121)
    imshow(abs(arrI), cmap="gray")
    
    subplot(122)
    imshow(abs(arrI_Reco), cmap="gray")
    
if nAx==3:
    figure()
    
    subplot(321)
    imshow(abs(arrI[nPix//2,:,:]), cmap="gray")
    subplot(322)
    imshow(abs(arrI_Reco[nPix//2,:,:]), cmap="gray")
    
    subplot(323)
    imshow(abs(arrI[:,nPix//2,:]), cmap="gray")
    subplot(324)
    imshow(abs(arrI_Reco[:,nPix//2,:]), cmap="gray")
    
    subplot(325)
    imshow(abs(arrI[:,:,nPix//2]), cmap="gray")
    subplot(326)
    imshow(abs(arrI_Reco[:,:,nPix//2]), cmap="gray")

# plot
figure(figsize=(18,9), dpi=120)

subplot(261)
plot(*lstArrK[0].T[(0,1),:], ".-")
axis("equal")
grid("on")
title(f"kx-ky {1}/{len(lstArrGrad)}")

if nAx==3:
    subplot(262)
    plot(*lstArrK[0].T[(0,2),:], ".-")
    axis("equal")
    grid("on")
    title("kx-kz")

    subplot(263)
    plot(*lstArrK[0].T[(1,2),:], ".-")
    axis("equal")
    grid("on")
    title("ky-kz")

subplot(222)
for iAx in range(nAx):
    plot(lstArrGrad[0][:,iAx]/(42.58e6)*(nPix/fov), ".-")
grid("on")
title("Gradient")

subplot(267)
plot(*lstArrGrad[0].T[(0,1),:], ".-")
axis("equal")
grid("on")
title("gx-gy")

if nAx==3:
    subplot(268)
    plot(*lstArrGrad[0].T[(0,2),:], ".-")
    axis("equal")
    grid("on")
    title("gx-gz")

    subplot(269)
    plot(*lstArrGrad[0].T[(1,2),:], ".-")
    axis("equal")
    grid("on")
    title("gy-gz")

subplot(224)
plot(norm(arrSlew,axis=-1)/(42.58e6)*(nPix/fov), ".-")
ylim(sLim/(42.58e6)*(nPix/fov)*0.9, sLim/(42.58e6)*(nPix/fov)*1.1)
grid("on")
title(f"Slewrate, max:{max(norm(arrSlew,axis=-1))/(42.58e6)*(nPix/fov):.3f}")

subplots_adjust(0.05,0.1,0.95,0.9, 0.2, 0.2)

show()
