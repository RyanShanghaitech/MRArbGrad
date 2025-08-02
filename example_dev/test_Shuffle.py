from numpy import *
from matplotlib.pyplot import *
from time import time

import math

GOLDRAT = 1.61803398875  # Golden ratio constant

def gen_rand_idx(pvlIdx, lN):
    llSeq = list(range(lN))
    illSeq = 0  # Python index instead of iterator

    lIntv = int(round(len(llSeq) * 1.0 / (GOLDRAT + 1.0)))
    for i in range(lN):
        for j in range(lIntv):
            illSeq += 1
            if illSeq == len(llSeq):
                illSeq = 0
        pvlIdx[i] = llSeq[illSeq]
        del llSeq[illSeq]
        if len(llSeq) == 0:
            break
        lIntv = int(round(len(llSeq) * 1.0 / (GOLDRAT + 1.0)))
        if illSeq == len(llSeq):
            illSeq = 0  # If last item removed, reset index

    return True

N = 65536
indices = [0] * N

t = time()
gen_rand_idx(indices, N)
t = time() - t
print(t)

arrIdx = array(indices)

# figure()
# plot(arrIdx, ".-")
# ylim([-1,N])
# show()