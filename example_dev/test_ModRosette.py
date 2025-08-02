from numpy import *
from matplotlib.pyplot import *

t = linspace(0,1,1000)
om1 = pi
om2_1 = pi
om2_2 = 16*pi

rho = 0.5*sin(om1*t)
x = rho*cos(amax([om2_1*t, om2_1*0.5+om2_2*(t-0.5)], axis=0))
y = rho*sin(amax([om2_1*t, om2_1*0.5+om2_2*(t-0.5)], axis=0))

figure()
plot(x, y, ".-")
axis("equal")
show()