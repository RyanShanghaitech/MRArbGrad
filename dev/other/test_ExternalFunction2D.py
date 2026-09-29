from numpy import *
from numpy.linalg import norm
from matplotlib.pyplot import *
import mrarbgrad as mag

gamma = 42.5756e6
fov = 0.256
nPix = 256
dtGrad = 10e-6
dtADC = 1e-6
sLim = 50 * gamma * fov/nPix
gLim = 20e-3 * gamma * fov/nPix
# gLim = 1/nPix/dtADC

# Rosette
kRhoPhi = 0.5/(2*pi)
plim = mag.trajfunc.prange_Yarnball(kRhoPhi)

lstGradLen = []
for tht0 in linspace(0,2*pi,100):
    trajfunc = lambda p: mag.trajfunc.Yarnball(p, kRhoPhi, tht0, 0)
    arrG = mag.calGrad4ExFunc(fov, nPix, sLim, gLim, dtGrad, trajfunc, p0=plim[0], p1=plim[1])[0]
    lstGradLen.append(arrG.shape[0])

figure()
plot(lstGradLen, ".-")
show()