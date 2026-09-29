from numpy import *
import random as py_random
from numpy.linalg import norm
from scipy.stats import qmc
from matplotlib.pyplot import *
import mrautograd as mag
from scipy.io import savemat, loadmat

gamma = 42.5756e6
fov = 0.256
nPix = 256
sLim = 100 * gamma * fov/nPix
gLim = 120e-3 * gamma * fov/nPix
dtGrad = 10e-6
dtADC = dtGrad

s = 42
py_random.seed(s)
random.seed(s)

# --- TSP Configuration and Pre-computation ---
nCity = 256*1

if 1:
    arrCity = mag.Utility.genTspTraj(nCity)
    arrCity = mag.Utility.rmCity(arrCity, pi/6, 1/nPix)
    # arrCity = mag.Utility.intpCity(arrCity, nPix)
    savemat("/mnt/d/LProject/minTimeGradient/Matlab/arrCity.mat", {"arrCity":arrCity})
else:
    arrCity = loadmat("/mnt/d/LProject/minTimeGradient/Matlab/arrCity.mat")["arrCity"]

# derive slew-rate constrained trajectory
nAx = 2
mag.setSolverMtg(0)
# mag.setMagOverSamp(8)
arrG, _ = mag.calGrad4ExSamp(False, fov, nPix, sLim, gLim, dtGrad, arrCity)
nRO, _ = arrG.shape

arrS = diff(arrG, axis=0)/dtGrad
print(f"sMax: {max(norm(arrS,axis=-1))/(42.58e6)*(nPix/fov)}")

arrK, _ = mag.cvtGrad2Traj(arrG, dtGrad, dtADC)
arrK += arrCity[:1,:]

# plot
figure(figsize=(20,10), dpi=120)

subplot(221, projection=None if nAx==2 else "3d")
if nAx==2: plot(arrK[:,0], arrK[:,1], ".-", label="K_Imp (TSP)")
if nAx==3: plot(arrK[:,0], arrK[:,1], arrK[:,2], ".-", label="K_Imp (TSP)")
# Also plot the discrete TSP points themselves
plot(arrCity[:,0], arrCity[:,1], 'o', markersize=8, color='red', label='TSP Cities', zorder=-1)
xlim(-0.5, 0.5)
ylim(-0.5, 0.5)
axis("equal")
grid("on")
legend()
title("k-Space (TSP Trajectory)")

subplot(222)
for iAx in range(nAx):
    plot(arrG[:,iAx]/(42.58e6)*(nPix/fov), ".-")
grid("on")
title("Gradient (TSP-derived)")

subplot(223, projection=None if nAx==2 else "3d")
if nAx==2: plot(arrG[:,0], arrG[:,1], ".-")
if nAx==3: plot(arrG[:,0], arrG[:,1], arrG[:,2], ".-")
axis("equal")
grid("on")
title("g-Space (TSP-derived)")

subplot(224)
plot(norm(arrS,axis=-1)/(42.58e6)*(nPix/fov), ".-")
ylim(sLim/(42.58e6)*(nPix/fov)*0.9, sLim/(42.58e6)*(nPix/fov)*1.1)
grid("on")
title(f"Slewrate, max:{max(norm(arrS,axis=-1))/(42.58e6)*(nPix/fov):.3f}")

show()