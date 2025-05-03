from numpy import *
from matplotlib.pyplot import *
from numpy.random import uniform
from scipy.stats.qmc import Sobol, Halton
import mrautograd as mag

Npt = 1000

# Sobel Sequence
sob = Sobol(2, scramble=False)
arrPtSobel = sob.random(Npt)

# Halton Sequence
hal = Halton(2, scramble=False)
arrPtHalton = hal.random(Npt)

# Uniform Distribution
arrPtUnif = uniform(0,1,(Npt,2))

# Cluster Distribution
arrPtClus = uniform(0.4,0.6,(Npt,2))

# Calculate Diaphony
diaSobel = g4n.calDiaphony(arrPtSobel)
diaHalton = g4n.calDiaphony(arrPtHalton)
diaUnif = g4n.calDiaphony(arrPtUnif)
diaClus = g4n.calDiaphony(arrPtClus)

print("diaSobel", diaSobel)
print("diaHalton", diaHalton)
print("diaUnif", diaUnif)
print("diaClus", diaClus)

# Plot
figure(figsize=(6,6), dpi=120)

subplot(221)
scatter(*arrPtSobel.T, s=1)
xlim(0,1)
ylim(0,1)
title(f"Sobel, Diaphony: {diaSobel:.2e}")

subplot(222)
scatter(*arrPtHalton.T, s=1)
xlim(0,1)
ylim(0,1)
title(f"Halton, Diaphony: {diaHalton:.2e}")

subplot(223)
scatter(*arrPtUnif.T, s=1)
xlim(0,1)
ylim(0,1)
title(f"Uniform, Diaphony: {diaUnif:.2e}")

subplot(224)
scatter(*arrPtClus.T, s=1)
xlim(0,1)
ylim(0,1)
title(f"Cluster, Diaphony: {diaClus:.2e}")

subplots_adjust(0.1,0.1,0.9,0.9,0.5,0.5)
show()