from numpy import *
from numpy.linalg import norm
from matplotlib.pyplot import *
import mrarbgrad as mag

gamma = 42.5756e6
fov = 0.256
nPix = 256
dtGrad = 10e-6 * 1e-1
dtADC = 2.5e-6 * 1e-1
sLim = 50 * gamma * fov/nPix
gLim = 20e-3 * gamma * fov/nPix
# gLim = 1/nPix/dtADC

# Rosette
nAx = 2
om1 = 19*pi
om2 = 17*pi
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
figure(figsize=(20,10), dpi=120)

subplot(221, projection=None if nAx==2 else "3d")
if nAx==2: plot(arrK_Ref[:,0], arrK_Ref[:,1], "--", label="K_Ref")
if nAx==3: plot(arrK_Ref[:,0], arrK_Ref[:,1], arrK_Ref[:,2], "--", label="K_Ref")
if nAx==2: plot(arrK[:,0], arrK[:,1], ".-", label="K_Imp")
if nAx==3: plot(arrK[:,0], arrK[:,1], arrK[:,2], ".-", label="K_Imp")
xlim(-0.5,0.5)
ylim(-0.5,0.5)
axis("equal")
grid("on")
legend()
title("k-Space")

subplot(222)
for iAx in range(nAx):
    plot(arrGrad[:,iAx]/(42.58e6)*(nPix/fov), ".-")
grid("on")
title("Gradient")

subplot(223, projection=None if nAx==2 else "3d")
if nAx==2: plot(arrGrad[:,0], arrGrad[:,1], ".-")
if nAx==3: plot(arrGrad[:,0], arrGrad[:,1], arrGrad[:,2], ".-")
axis("equal")
grid("on")
title("g-Space")

subplot(224)
plot(norm(arrSlew,axis=-1)/(42.58e6)*(nPix/fov), ".-")
ylim(sLim/(42.58e6)*(nPix/fov)*0.9, sLim/(42.58e6)*(nPix/fov)*1.1)
grid("on")
title(f"Slewrate, max:{max(norm(arrSlew,axis=-1))/(42.58e6)*(nPix/fov):.3f}")

show()
