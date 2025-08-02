import mrautograd as mag
from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
from time import time
import finufft as fn
import slime
import fars

enSim = 0
gamma = 42.5756e6
# fov = 0.384
fov = 0.256
# fov = 0.192
nPix = 256
sLim = 50 * gamma * fov/nPix
gLim = 20e-3 * gamma * fov/nPix
dtGrad = 10e-6
dtADC = 5e-6
argCom = dict(dFov=fov, lNPix=nPix, dSLim=sLim, dGLim=gLim, dDt=dtGrad)

mag.ext.setSolverMtg(0)
mag.ext.setTrajRev(0)
mag.ext.setGoldAng(0)
mag.ext.setMaxG0(0)
mag.ext.setMaxG1(0)

# calculate gradient
t = time()
for i in range(1):
    # lstArrK0, lstArrGrad = mag.Function.getG_Spiral(bIs3D=0, **argCom); nAx = 2 # 0.380s
    # lstArrK0, lstArrGrad = mag.Function.getG_VarDenSpiral(bIs3D=0, **argCom); nAx = 2 # 0.499s
    lstArrK0, lstArrGrad = mag.Function.getG_Rosette(bIs3D=0, **argCom); nAx = 2 # 16.39s (9 frames)
    # lstArrK0, lstArrGrad = mag.Function.getG_Rosette_Trad(**argCom, dOm1=10*pi, dOm2=8*pi, dTmax=1, dTacq=2e-03); nAx = 2 # 16.39s (9 frames)
    # lstArrK0, lstArrGrad = mag.Function.getG_Shell3d(dRhoTht=0.5/(2*pi), **argCom); nAx = 3 # 183.7
    # lstArrK0, lstArrGrad = mag.Function.getG_Yarnball(dRhoPhi=0.5/(2*pi), **argCom); nAx = 3 # 196.1
    # lstArrK0, lstArrGrad = mag.Function.getG_Seiffert(**argCom); nAx = 3 # 232.9s
    # lstArrK0, lstArrGrad = mag.Function.getG_Cones(**argCom); nAx = 3 # 149.9s
t = time() - t
print(f"Exe. time: {t*1e3:.3f} ms")

# exit()

# print(f"Intlea Num.: {len(lstArrGrad)}")

lstArrGrad_Del = lstArrGrad.copy()
for i in range(len(lstArrGrad_Del)):
    arrG_r0 = lstArrGrad_Del[i]
    arrG_r1 = roll(lstArrGrad_Del[i], (1,), 0); arrG_r1[:1,:]*=0
    arrG_r2 = roll(lstArrGrad_Del[i], (2,), 0); arrG_r2[:2,:]*=0
    arrG_r3 = roll(lstArrGrad_Del[i], (3,), 0); arrG_r3[:3,:]*=0
    lstArrGrad_Del[i] = arrG_r0*1 + arrG_r1*0 + arrG_r2*0 + arrG_r3*0

nRO_Max = max(arrG.shape[0] for arrG in lstArrGrad)
# print(f"Tacq: {nRO_Max*dtGrad*1e3:.3f} ms")
tTR = (nRO_Max*dtGrad + 2e-3)
# print(f"TR: {tTR*1e3:.3f} ms")
tScan = tTR*len(lstArrGrad)
# print(f"Tscan: {tScan:.3e} s")

# derive shape parameter
if nAx==2:
    lstArrK0 = [arrK0[:2] for arrK0 in lstArrK0]
    lstArrGrad = [arrG[:,:2] for arrG in lstArrGrad]
    lstArrGrad_Del = [arrG[:,:2] for arrG in lstArrGrad_Del]
nRO, nAx = lstArrGrad[0].shape

# derive slewrate
lstArrSlew = [diff(arrG, axis=0)/dtGrad for arrG in lstArrGrad]
sMax = max(norm(concatenate(lstArrSlew)/gamma*nPix/fov,axis=-1))
gMax = max(norm(concatenate(lstArrGrad)/gamma*nPix/fov,axis=-1))
# print(f"sMax: {sMax}")
# print(f"gMax: {gMax}")
print(f"overshoot: {(sMax-sLim/gamma*nPix/fov)/(sLim/gamma*nPix/fov)*100:.3f}%")

# derive trajectory
lstArrK = []
for arrK0, arrGrad in zip(lstArrK0, lstArrGrad):
    arrK, _ = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
    arrK += arrK0
    lstArrK.append(arrK)
    
lstArrK_Del = []
for arrK0, arrGrad in zip(lstArrK0, lstArrGrad_Del):
    arrK, _ = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
    arrK += arrK0
    lstArrK_Del.append(arrK)

# # plot
# if nAx==3:
#     figure(figsize=(9,9), dpi=120)
#     subplot(111, projection="3d")
#     plot(*array([arrK[-1,:] for arrK in lstArrK[-100:]]).T, ".", linestyle='')
#     title("last 100 intlea.")

# interleaf to be plotted
# iArrK = argmax(array([amax(norm(arrS,axis=-1)) for arrS in lstArrSlew]))
iArrK = 0

# k-space and g-space
figure(figsize=(18,9), dpi=120)

subplot(221, projection="3d" if nAx==3 else None)
plot(*lstArrK[iArrK].T, ".-")
axis("equal")
grid("on")
title(f"kspace {1}/{len(lstArrGrad)}")

subplot(223, projection="3d" if nAx==3 else None)
plot(*lstArrGrad[iArrK].T, ".-")
axis("equal")
grid("on")
title("gspace")

# gradient and slewrate
subplot(222)
for iAx in range(nAx):
    plot(lstArrGrad[iArrK][:,iAx]/gamma*nPix/fov, ".-")
grid("on")
title(f"Gradient")

subplot(224)
plot(norm(lstArrSlew[iArrK],axis=-1)/gamma*nPix/fov, ".-", c="tab:blue")
ylim(sLim/gamma*nPix/fov*0.9, sLim/gamma*nPix/fov*1.1)
grid("on")

twinx()
plot(norm(lstArrGrad[iArrK],axis=-1)/gamma*nPix/fov*1e3, ".-", c="tab:orange")
ylim(gLim/gamma*nPix/fov*0.9*1e3, gLim/gamma*nPix/fov*1.1*1e3)
grid("on")
title(f"Grad & Slew amp., max grad:{gMax*1e3:.3f}, max slew:{sMax:.3f}")

subplots_adjust(0.05,0.1,0.95,0.9, 0.2, 0.2)

figure()
ax = subplot(111, projection="3d")
arrK = mag.ext.getTestVal()
plot(arrK[:,0], arrK[:,1], arrK[:,2], ".-")
ax.set_xlim([-0.5,0.5])
ax.set_ylim([-0.5,0.5])
ax.set_zlim([-0.5,0.5])

if not enSim:
    show()
    exit(0)



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
    figure(figsize=(9,5), dpi=120)
    
    subplot(121)
    imshow(abs(arrI), cmap="gray")
    colorbar()
    
    subplot(122)
    imshow(abs(arrI_Reco), cmap="gray")
    colorbar()
    
if nAx==3:
    figure(figsize=(6,9), dpi=120)
    
    subplot(321)
    imshow(abs(arrI[nPix//2,:,:]), cmap="gray")
    colorbar()
    subplot(322)
    imshow(abs(arrI_Reco[nPix//2,:,:]), cmap="gray")
    colorbar()
    
    subplot(323)
    imshow(abs(arrI[:,nPix//2,:]), cmap="gray")
    colorbar()
    subplot(324)
    imshow(abs(arrI_Reco[:,nPix//2,:]), cmap="gray")
    colorbar()
    
    subplot(325)
    imshow(abs(arrI[:,:,nPix//2]), cmap="gray")
    colorbar()
    subplot(326)
    imshow(abs(arrI_Reco[:,:,nPix//2]), cmap="gray")
    colorbar()
    
show()
