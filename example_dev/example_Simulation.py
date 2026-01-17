import mrarbgrad as mag
from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import finufft as fn
from mrphantom import *
import mrarbdcf as mad
from time import time

enSim = 1
gamma = 42.5756e6
fov = 0.256 # 0.256
nPix = 256
dtGrad = 10e-6
dtADC = 2.5e-6
sLim = 100 * gamma * fov/nPix
gLim = 1/nPix/dtADC #  120e-3 * gamma * fov/nPix
argCom = dict(fov=fov, nPix=nPix, sLim=sLim, gLim=gLim, dt=dtGrad)

mag.setSolverMtg(0)
mag.setTrajRev(0)
mag.setGoldAng(0)
mag.setShuf(0)
mag.setMaxG0(0)
mag.setMaxG1(0)
# mag.setMagOverSamp(8)
mag.setMagSFS(0)
mag.setMagGradRep(1)
mag.setMagTrajRep(1)
mag.setDbgPrint(1)

# calculate gradient
# lstArrK0, lstArrGrad = mag.getG_Spiral(**argCom); nAx = 2
# lstArrK0, lstArrGrad = mag.getG_VDSpiral(**argCom); nAx = 2
# lstArrK0, lstArrGrad = mag.getG_VDSpiral_RT(**argCom); nAx = 2
# lstArrK0, lstArrGrad = mag.getG_Rosette(**argCom); nAx = 2
# lstArrK0, lstArrGrad = mag.getG_Rosette_Trad(**argCom, om1=10*pi, om2=8*pi, tMax=1, tAcq=2e-03); nAx = 2
# lstArrK0, lstArrGrad = mag.getG_Shell3d(**argCom, kRhoTht = 0.5 / (2 * pi)); nAx = 3
# lstArrK0, lstArrGrad = mag.getG_Yarnball(**argCom, kRhoPhi = 0.5 / (2 * pi)); nAx = 3
lstArrK0, lstArrGrad = mag.getG_Yarnball_RT(**argCom, kRhoPhi = 0.5 / (2 * pi)); nAx = 3
# lstArrK0, lstArrGrad = mag.getG_Seiffert(**argCom); nAx = 3
# lstArrK0, lstArrGrad = mag.getG_Cones(**argCom, kRhoPhi = 0.5 / (16 * pi)); nAx = 3

# lstArrGrad = mag.gradClip(lstArrGrad, dtGrad, sLim, gLim)

print(f"Intlea Num.: {len(lstArrGrad)}")
nRO_Max = amax([arrG.shape[0] for arrG in lstArrGrad])
print(f"Tacq: {nRO_Max*dtGrad*1e3:.3f} ms")
tTR = (nRO_Max*dtGrad + 2e-3)
print(f"TR: {tTR*1e3:.3f} ms")
tScan = tTR*len(lstArrGrad)
print(f"Tscan: {tScan:.3e} s")

# exit()

# derive shape parameter
if nAx==2:
    lstArrK0 = [arrK0[:2] for arrK0 in lstArrK0]
    lstArrGrad = [arrG[:,:2] for arrG in lstArrGrad]
nRO, nAx = lstArrGrad[0].shape

# derive slewrate
lstArrSlew = [diff(arrG, axis=0)/dtGrad for arrG in lstArrGrad]
sMax = max(norm(concatenate(lstArrSlew)/gamma*nPix/fov, axis=-1))
gMax = max(norm(concatenate(lstArrGrad)*1e3/gamma*nPix/fov, axis=-1))
print(f"sMax: {sMax:.3f} T/m/s")
print(f"gMax: {gMax:.3f} mT/m")

# derive trajectory
lstArrK = []
for arrK0, arrGrad in zip(lstArrK0, lstArrGrad):
    arrK, _ = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
    arrK += arrK0
    lstArrK.append(arrK)

# simulate phantom
arrI = Enum2M0(genPhant(nAx, nPix)).squeeze()
arrK = concatenate(lstArrK, axis=0)

mad.setNumStep(2)
t = time()
lstArrDcf = mad.sovDcf(nPix, lstArrK, sWind="cos", pShape=1.0)
t = time() - t
arrDcf = hstack(lstArrDcf).astype(complex64)
print(f"sovDcf: {t:.2f} s")

arr2PiKT = asarray(2*pi*arrK.T, dtype=float32, order="C")

plan = fn.Plan(2, tuple(nPix for _ in range(nAx)), isign=-1, dtype="complex64", showwarn=0)
plan.setpts(*arr2PiKT)
arrS = plan.execute(arrI.astype(complex64))

plan = fn.Plan(1, tuple(nPix for _ in range(nAx)), isign=1, dtype="complex64", showwarn=0)
plan.setpts(*arr2PiKT)
arrI_Reco = plan.execute(arrS*arrDcf)

figure(figsize=(4,6), dpi=120)
subplot(211)
for i in range(nAx):
    plot(lstArrGrad[0][:,i], ".-")
subplot(212)
for i in range(nAx):
    plot(norm(lstArrGrad[0], axis=-1)/gLim, ".-")
    plot(norm(lstArrSlew[0], axis=-1)/sLim, ".-")

if nAx==2:
    figure(figsize=(9,5), dpi=120)
    
    subplot(121)
    imshow(abs(arrI), cmap="gray")
    clim(0,1)
    colorbar()
    
    subplot(122)
    imshow(abs(arrI_Reco), cmap="gray")
    clim(0,1)
    colorbar()
    
if nAx==3:
    figure(figsize=(6,9), dpi=120)
    
    subplot(321)
    imshow(abs(arrI[nPix//2,:,:]), cmap="gray")
    clim(0,1)
    colorbar()
    subplot(322)
    imshow(abs(arrI_Reco[nPix//2,:,:]), cmap="gray")
    clim(0,1)
    colorbar()
    
    subplot(323)
    imshow(abs(arrI[:,nPix//2,:]), cmap="gray")
    clim(0,1)
    colorbar()
    subplot(324)
    imshow(abs(arrI_Reco[:,nPix//2,:]), cmap="gray")
    clim(0,1)
    colorbar()
    
    subplot(325)
    imshow(abs(arrI[:,:,nPix//2]), cmap="gray")
    clim(0,1)
    colorbar()
    subplot(326)
    imshow(abs(arrI_Reco[:,:,nPix//2]), cmap="gray")
    clim(0,1)
    colorbar()
    
show()
