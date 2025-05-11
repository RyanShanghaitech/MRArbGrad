import mrautograd as mag
from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
from time import time
import finufft as fn
import slime
import fars
import torch as tor
import torchkbnufft as tkbn

gamma = 42.5756e6
# fov = 0.384
fov = 0.256
nPix = 256
sLim = 100 * gamma * fov/nPix
gLim = 120e-3 * gamma * fov/nPix
dtGrad = 10e-6
dtADC = 5e-6
argCom = dict(dFov=fov, lNPix=nPix, dSLim=sLim, dGLim=gLim, dDt=dtGrad)

enSim = 1

# calculate gradient
t = time()

# lstArrK0, lstArrGrad = mag.Function.getG_Spiral(bIs3D=1, **argCom); nAx = 3 # 0.380s

# lstArrK0, lstArrGrad = mag.Function.getG_VarDenSpiral(bIs3D=1, **argCom, dRhoPhi0=0.5/(4*pi), dRhoPhi1=0.5/(32*pi)); nAx = 3 # 0.499s

# lstArrK0, lstArrGrad = mag.Function.getG_Rosette(**argCom, dOm1=5*pi, dOm2=3*pi, dTmax=1); nAx = 2 # 16.39s (9 frames)

tAcq_ms = 8.66; lstArrK0, lstArrGrad = mag.Function.getG_Rosette_Trad(**argCom, dOm1=5*pi/tAcq_ms, dOm2=3*pi/tAcq_ms, dTmax=1*tAcq_ms); nAx = 2 # 16.39s (9 frames)

# lstArrK0, lstArrGrad = mag.Function.getG_Shell3d(dRhoTht=0.5/(2*pi), **argCom); nAx = 3 # 183.7

# lstArrK0, lstArrGrad = mag.Function.getG_Yarnball(dRhoPhi=0.5/(2*pi), **argCom); nAx = 3 # 196.1
    
# lstArrK0, lstArrGrad = mag.Function.getG_Seiffert(**argCom); nAx = 3 # 232.9s

# lstArrK0, lstArrGrad = mag.Function.getG_Cones(**argCom); nAx = 3 # 149.9s

t = time() - t
print(f"Exe Time: {t}")
print(f"Intlea Num.: {len(lstArrGrad)}")

if all(array(lstArrK0)==0): print("NO PE")

nRO_Max = max(arrG.shape[0] for arrG in lstArrGrad)
print(f"Tacq: {nRO_Max*dtGrad*1e3:.3f} ms")
tTR = (nRO_Max*dtGrad + 6e-3)
print(f"TR: {tTR*1e3:.3f} ms")
tScan = tTR*len(lstArrGrad)
print(f"Tscan: {tScan:.3e} s")

# derive shape parameter
if nAx==2:
    lstArrGrad = [arrG[:,:2] for arrG in lstArrGrad]
    lstArrK0 = [arrK0[:2] for arrK0 in lstArrK0]
nRO, nAx = lstArrGrad[0].shape

# derive slewrate
lstArrSlew = [diff(arrG, axis=0)/dtGrad for arrG in lstArrGrad]
sMax = max(norm(concatenate(lstArrSlew)/gamma*nPix/fov,axis=-1))
gMax = max(norm(concatenate(lstArrGrad)/gamma*nPix/fov,axis=-1))
print(f"sMax: {sMax}")
print(f"gMax: {gMax}")

# derive trajectory
lstArrK = []
for arrK0, arrGrad in zip(lstArrK0, lstArrGrad):
    arrK = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
    arrK += arrK0
    lstArrK.append(arrK)

# # plot
# if nAx==3:
#     figure(figsize=(9,9), dpi=120)
#     subplot(111, projection="3d")
#     plot(*array([arrK[-1,:] for arrK in lstArrK[-100:]]).T, ".", linestyle='')
#     title("last 100 intlea.")

# interleaf to be plotted
iArrK = argmax(array([amax(norm(arrS,axis=-1)) for arrS in lstArrSlew]))
# iArrK = 0

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
title(f"Gradient, max:{gMax*1e3:.3f}")

subplot(224)
plot(norm(lstArrSlew[iArrK],axis=-1)/gamma*nPix/fov, ".-")
ylim(sLim/gamma*nPix/fov*0.9, sLim/gamma*nPix/fov*1.1)
grid("on")
title(f"Slewrate, max:{sMax:.3f}")

subplots_adjust(0.05,0.1,0.95,0.9, 0.2, 0.2)


if not enSim:
    show()
    exit(0)



# simulate phantom
arrI = slime.genPhan(nAx, nPix)["M0"].squeeze()
# if nAx == 2: arrI = arrI[nPix//2,:,:]
arrK = concatenate(lstArrK, axis=0)

arrDcf = fars.calDcf(nPix, arrK[:,:nAx]).astype(complex64)

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
