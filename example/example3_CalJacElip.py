from numpy import *
from matplotlib.pyplot import *
from scipy.special import ellipj
import g4n

arrU = linspace(0, 8*pi/2, 1002)[1:-1]
m = 0.1

# AGM
arrSn, arrCn = g4n.calJacElip(arrU, m)

# scipy implementation
arrSn_Ref, arrCn_Ref, _, _ = ellipj(arrU, m)

fig = figure(figsize=(6,6), dpi=120)

ax = fig.add_subplot(211)
ax.plot(arrU, arrSn, ".-", label="derived")
ax.plot(arrU, arrSn_Ref, ".-", label="reference")
ax.legend()
ax.set_title("sn")

ax = fig.add_subplot(212)
ax.plot(arrU, arrCn, ".-", label="derived")
ax.plot(arrU, arrCn_Ref, ".-", label="reference")
ax.legend()
ax.set_title("cn")

show()