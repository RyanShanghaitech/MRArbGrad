from numpy import *
import mrarbgrad as mag

def Rosette(t):
    rho = 0.5 * sin(5*pi*t)
    phi = 3*pi*t
    return rho*array([cos(phi), sin(phi)])

grad = mag.solve(Rosette, pmin=0, pmax=1)

# README CODE END
from matplotlib.pyplot import *

figure()
for i in range(grad.shape[1]):
    plot(grad[:,i], ".-")
savefig(__file__.replace(".py", "_fig.png"), dpi=150)
