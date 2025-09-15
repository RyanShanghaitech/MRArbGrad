import mrautograd as mag
from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import finufft as fn
import slime
import fars

enSim = 1
gamma = 42.5756e6
fov = 0.320 # 0.256
nPix = 256
sLim = 100 * gamma * fov/nPix
gLim = 120e-3 * gamma * fov/nPix
dtGrad = 10e-6
dtADC = 2.5e-6
argCom = dict(dFov=fov, lNPix=nPix, dSLim=sLim, dGLim=gLim, dDt=dtGrad)

mag.setSolverMtg(0)
mag.setTrajRev(0)
mag.setGoldAng(1)
mag.setMaxG0(0)
mag.setMaxG1(0)
mag.setExGEnd(0)
mag.setMagOv(8)

# calculate gradient
# lstArrK0, lstArrGrad = mag.getG_Spiral(bIs3D=0, **argCom); nAx = 2
# lstArrK0, lstArrGrad = mag.getG_VarDenSpiral(bIs3D=0, **argCom, dRhoPhi0 = 0.5 / (256 * pi), dRhoPhi1 = 0.5 / (12 * pi)); nAx = 2
# lstArrK0, lstArrGrad = mag.getG_Rosette(bIs3D=0, **argCom); nAx = 2
# lstArrK0, lstArrGrad = mag.getG_Rosette_Trad(**argCom, dOm1=10*pi, dOm2=8*pi, dTmax=1, dTacq=2e-03); nAx = 2
# lstArrK0, lstArrGrad = mag.getG_Shell3d(**argCom, dRhoTht = 0.5 / (4 * pi)); nAx = 3
lstArrK0, lstArrGrad = mag.getG_Yarnball(**argCom, dRhoPhi = 0.5 / (3 * pi)); nAx = 3
# lstArrK0, lstArrGrad = mag.getG_Seiffert(**argCom); nAx = 3
# lstArrK0, lstArrGrad = mag.getG_Cones(**argCom, dRhoPhi = 0.5 / (16 * pi)); nAx = 3

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
gMax = max(norm(concatenate(lstArrGrad)/gamma*nPix/fov, axis=-1))

# derive trajectory
lstArrK = []
for arrK0, arrGrad in zip(lstArrK0, lstArrGrad):
    arrK, _ = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC)
    arrK += arrK0
    lstArrK.append(arrK)

# simulate phantom
print(0); arrI = slime.genPhan(nAx, nPix)["M0"].squeeze(); print(1) # test
arrK = concatenate(lstArrK, axis=0)

arrDcf = fars.calDcf(nPix, arrK[:,:nAx]).astype(complex64)

arrOm = 2*pi*arrK; arrOm = arrOm.astype(float32)

plan = fn.Plan(2, tuple(nPix for _ in range(nAx)), isign=-1, dtype="complex64")
plan.setpts(*arrOm.T)
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
