import mrautograd as mag
from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
from time import time
import finufft as fn
import slime
import fars

gamma = 42.5756e6
fov = 0.256
nPix = 256
sLim = 100 * gamma * fov/nPix
gLim = 120e-3 * gamma * fov/nPix
dtGrad = 10e-6
dtADC = 2.5e-6
argCom = dict(dFov=fov, lNPix=nPix, dSLim=sLim, dGLim=gLim, dDt=dtGrad)

# lstArrK0, lstArrGrad = mag.Function.getG_Spiral(bIs3D=0, **argCom); nAx = 2 # 0.380s
lstArrK0, lstArrGrad = mag.Function.getG_Yarnball(dRhoPhi=0.5/(4*pi), **argCom); nAx = 3 # 196.1

lstArrGrad_Del = []
for i in range(len(lstArrGrad)):
    if i%100==0: print(i)
    lstArrGrad_Del.append(mag.Utility.delGrad(lstArrGrad[i], 0.5))

# derive trajectory
lstArrK = []
for arrK0, arrGrad in zip(lstArrK0, lstArrGrad):
    arrK = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
    arrK += arrK0
    lstArrK.append(arrK)
    
lstArrK_Del = []
for arrK0, arrGrad in zip(lstArrK0, lstArrGrad_Del):
    arrK = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
    arrK += arrK0
    lstArrK_Del.append(arrK)

# interleaf to be plotted
iArrK = 0

# k-space and g-space
figure(figsize=(18,9), dpi=120)

subplot(121, projection="3d" if nAx==3 else None)
plot(*lstArrK[iArrK].T[:nAx,:], ".-")
axis("equal")
grid("on")
title(f"kspace {1}/{len(lstArrGrad)}")

# gradient
subplot(222)
for iAx in range(nAx):
    plot(lstArrGrad[iArrK][:,iAx]/gamma*nPix/fov, ".-")
grid("on")

# gradient with delay
subplot(224)
for iAx in range(nAx):
    plot(lstArrGrad_Del[iArrK][:,iAx]/gamma*nPix/fov, ".-")
grid("on")

subplots_adjust(0.05,0.1,0.95,0.9, 0.2, 0.2)

# show()
# exit()


# simulate phantom
arrI = slime.genPhan(nAx, nPix)["M0"].squeeze()
arrK = concatenate(lstArrK, axis=0)
arrK_Del = concatenate(lstArrK_Del, axis=0)

arrDcf = fars.calDcf(nPix, arrK[:,:nAx]).astype(complex64)

arrOm = 2*pi*arrK; arrOm = arrOm.astype(float32)
arrOm_Del = 2*pi*arrK_Del; arrOm_Del = arrOm_Del.astype(float32)

plan = fn.Plan(2, tuple(nPix for _ in range(nAx)), isign=-1, dtype="complex64")
plan.setpts(*arrOm_Del.T)
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
