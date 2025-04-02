import g4n
from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
from time import time
import finufft as fn
import fars
import torch as tor
import torchkbnufft as tkbn

fov = 0.25
nPix = 256
sLim = 100 * 42.5756e6 * fov/nPix
gLim = 120e-3 * 42.5756e6 * fov/nPix
dtGrad = 10e-6
dtADC = 2.5e-6
nAx = 3
argCom = dict(lNPix=nPix, dSLim=sLim, dGLim=gLim, dDt=dtGrad)

# calculate gradient
t = time()

# lstArrK0, lstArrGrad = g4n.Function.getG_Spiral(lNStack=256, **argCom) # 0.380s

# lstArrK0, lstArrGrad = g4n.Function.getG_VarDenSpiral(lNStack=256, **argCom) # 0.499s

# lstArrK0, lstArrGrad = g4n.Function.getG_Rosette(lNStack=1, **argCom); nAx = 2 # 16.39s (9 frames)

# lstArrK0, lstArrGrad = g4n.Function.getG_CloseSpiral(lNStack=256, **argCom) # 0.462s

lstArrK0, lstArrGrad = g4n.Function.getG_Shell3d(**argCom) # 183.7

# lstArrK0, lstArrGrad = g4n.Function.getG_Yarnball(**argCom) # 196.1
    
# lstArrK0, lstArrGrad = g4n.Function.getG_Seiffert(**argCom) # 232.9s

# lstArrK0, lstArrGrad = g4n.Function.getG_Cones(**argCom) # 149.9s

t = time() - t
print(f"Exe Time: {t}")
print(f"Intlea Num.: {len(lstArrGrad)}")

nRO_Max = max(arrG.shape[0] for arrG in lstArrGrad)
tTR = (nRO_Max*dtGrad + 5e-3)
print(f"TR: {tTR*1e3:.3f} ms")
tScan = tTR*len(lstArrGrad)
print(f"Tscan: {tScan:.3e} s")

# derive shape parameter
if nAx==2:
    lstArrGrad = [arrG[:,:2] for arrG in lstArrGrad]
    lstArrK0 = [arrK0[:2] for arrK0 in lstArrK0]
nRO, nAx = lstArrGrad[0].shape

# derive slewrate
arrSlew = diff(lstArrGrad[0], axis=0)/dtGrad
print(f"sMax: {max(norm(arrSlew,axis=-1))/(42.58e6)*(nPix/fov)}")

# derive trajectory
lstArrK = []
for arrK0, arrGrad in zip(lstArrK0, lstArrGrad):
    arrK = g4n.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
    arrK += arrK0
    lstArrK.append(arrK)
    
_lst = []
for arrK in lstArrK:
    arrRho = norm(arrK, axis=-1)
    _lst.append(min(arrRho))
_arr = array(_lst)
print("mean", mean(_arr))
print("min", min(_arr))
print("max", max(_arr))

# plot
figure(figsize=(18,9), dpi=120)

iArrK = 64

subplot(261)
plot(*lstArrK[iArrK].T[(0,1),:], ".-")
axis("equal")
grid("on")
title(f"kx-ky {1}/{len(lstArrGrad)}")

if nAx==3:
    subplot(262)
    plot(*lstArrK[iArrK].T[(0,2),:], ".-")
    axis("equal")
    grid("on")
    title("kx-kz")

    subplot(263)
    plot(*lstArrK[iArrK].T[(1,2),:], ".-")
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



# show()
# exit(0)



# simulate phantom
arrI = asarray(load("./resource/arrM0.npz")["arrM0"])
if nAx == 2: arrI = arrI[nPix//2,:,:]
arrK = concatenate(lstArrK, axis=0)

arrDcf = fars.calDcf(nPix, arrK).astype(complex64)

# tenK = tor.from_numpy(arrK)
# tenDcf = tkbn.calc_density_compensation_function((2*pi)*tenK.T, (nPix,nPix), numpoints=8, kbwidth=4, num_iterations=1, table_oversamp=2**10)
# arrDcf = tenDcf.detach().cpu().numpy().squeeze().astype(complex64)

arrOm = 2*pi*arrK; arrOm = arrOm.astype(float32)

plan = fn.Plan(2, tuple(nPix for _ in range(nAx)), isign=-1, dtype="complex64")
plan.setpts(*arrOm.T)
arrS = plan.execute(arrI.astype(complex64))

plan = fn.Plan(1, tuple(nPix for _ in range(nAx)), isign=1, dtype="complex64")
plan.setpts(*arrOm.T)
arrI_Reco = plan.execute(arrS*arrDcf)

if nAx==2:
    figure(figsize=(9,9), dpi=120)
    
    subplot(121)
    imshow(abs(arrI), cmap="gray")
    
    subplot(122)
    imshow(abs(arrI_Reco), cmap="gray")
    
if nAx==3:
    figure(figsize=(9,9), dpi=120)
    
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
    
show()
