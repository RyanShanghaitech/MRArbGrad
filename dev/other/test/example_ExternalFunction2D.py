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
nAx = 2
om1 = 5*pi
om2 = 3*pi
def Rosette(t):
    rho = 0.5*sin(om1*t)
    return array\
    ([
        rho*cos(om2*t),
        rho*sin(om2*t),
        0*t
    ])
    
pLim = [0,1]

# derive slew-rate constrained trajectory
mag.setMaxG0(0)
mag.setMaxG1(0)
arrGrad = mag.calGrad4ExFunc(fov, nPix, sLim, gLim, dtGrad, Rosette, None, None, pLim[0], pLim[1])[0]
# arrGrad = mag.gradClip(arrGrad, dtGrad, sLim, gLim) # clip slew/grad amp with hardware constraint
nRO = arrGrad.shape[0]

arrSlew = diff(arrGrad, axis=0)/dtGrad
print(f"sMax: {max(norm(arrSlew,axis=-1))/(42.58e6)*(nPix/fov)}")

arrK = mag.cvtGrad2Traj(arrGrad, dtGrad, dtADC, 0.5)[0]

# derive reference trajectory
arrP_Ref = linspace(pLim[0], pLim[1], int(1e4))
arrK_Ref = Rosette(arrP_Ref).T

# plot
figure(figsize=(6,6), dpi=200)

subplot(211, projection=None if nAx==2 else "3d")
if nAx==2: plot(arrK[:,0], arrK[:,1], "-", linewidth=2, label="K_Imp")
if nAx==3: plot(arrK[:,0], arrK[:,1], arrK[:,2], "-", linewidth=2, label="K_Imp")
xlim(-0.5,0.5)
ylim(-0.5,0.5)
axis("equal")
grid("on")
title("k-Space")

subplot(212)
for iAx in range(nAx):
    y = arrGrad[:,iAx]/(42.58e6)*(nPix/fov)
    x = 1e3*dtGrad * arange(len(y))
    plot(x, y, "-", linewidth=2)
xlabel("t [ms]")
grid("on")
title("Gradient")

savefig("fig.svg")
# show()
