from numpy import *
from matplotlib.pyplot import *
from scipy.special import ellipj
import mrautograd as mag

arrU = linspace(0, 8*pi/2, 100002)[1:-1]
m = 0.07

# AGM
arrSn, arrCn = mag.calJacElip(arrU, m)
period = 4*mag.calCompElipInt(m)
print(period)
iPeriod = argmin(abs(arrU-period))
print(abs(arrSn[iPeriod]-arrSn[0]))
print(abs(arrCn[iPeriod]-arrCn[0]))

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