from numpy import *
from matplotlib.pyplot import *
from numpy.linalg import norm
import mrautograd as mag

# brutely search parameter u of Seiffert Spiral, depends on desired diaphony

arrM = linspace(0.01,0.99,99)
arrD = ones_like(arrM)
nM = arrM.shape[0]

nK = 1000
for iM in range(nM):
    print(f"{iM}/{nM}")
    m = arrM[iM]
    # uPrd = 4*g4n.calCompElipInt(m) # period of argument u
    arrU = linspace(0,20,nK)
    arrSn, arrCn = g4n.calJacElip(arrU, m)
    arrPhi = sqrt(m)*arrU
    arrK = 0.5*(linspace(0,1,nK)[:,newaxis]**2)*array([arrSn*cos(arrPhi), arrSn*sin(arrPhi), arrCn], dtype=float64).T
    
    # continue
    
    d = g4n.calDiaphony(arrK*(254/256)+1/2)
    arrD[iM] = d
    
iM_Optm = argmin(arrD)
m = arrM[iM_Optm]
arrU = linspace(0,20,nK)
arrSn, arrCn = g4n.calJacElip(arrU, m)
arrPhi = sqrt(m)*arrU
arrK = 0.5*(linspace(0,1,nK)[:,newaxis]**1)*array([arrSn*cos(arrPhi), arrSn*sin(arrPhi), arrCn], dtype=float64).T
d = g4n.calDiaphony(arrK*(254/256)+1/2)

# plot
fig = figure(figsize=(12,6), dpi=120)

ax = fig.add_subplot(121, projection="3d")
ax.clear()
ax.plot(arrK[:,0], arrK[:,1], arrK[:,2], ".-", markersize=1, linewidth=1/2)
ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.5,0.5)
ax.set_zlim(-0.5,0.5)
ax.axis("equal")
ax.set_title(f"m={m:.3f}, d={d:.3f}")

ax = fig.add_subplot(122)
ax.clear()
ax.plot(arrM, arrD, ".-")
xlabel("m")
ylabel("diaphony")
grid("on")

fig.subplots_adjust(0.1,0.2,0.9,0.9, 0.75, 0.5)

show()