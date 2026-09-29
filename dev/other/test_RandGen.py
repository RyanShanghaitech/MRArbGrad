py_max = max

from numpy import *
from numpy.typing import NDArray
from scipy.stats import qmc
from matplotlib.pyplot import *

def rand(i:int|NDArray, nAx:int=3, kx=sqrt(2), ky=sqrt(3), kz=sqrt(7)) -> NDArray:
    return (hstack if size(i)==1 else vstack)\
    ([
        (i**1 * 1/(1+kx))%1,
        (i**1 * kx/(1+kx))%1,
        (i**3 * 1/(1+kz))%1
    ][:nAx]).T
    
arrSamp = rand(arange(1000))
disc = py_max\
    (
        qmc.discrepancy(arrSamp[:,(0,1)]),
        qmc.discrepancy(arrSamp[:,(0,2)]),
        qmc.discrepancy(arrSamp[:,(1,2)])
    )
arrSamp = rand(arange(10000))[:,(0,1)]
disc = qmc.discrepancy(arrSamp)

fig = figure()
ax = fig.add_subplot(111)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_title(f"discrepancy: {disc:.3e}")
ax.scatter(*arrSamp.T, s=1, c="k")

fig = figure()
ax = fig.add_subplot(111)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_title(f"discrepancy: {disc:.3e}")
for samp in arrSamp:
    ax.scatter(*samp, c="k")
    draw()
    pause(1e-3)