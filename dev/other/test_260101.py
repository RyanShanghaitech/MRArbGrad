import numpy as np
from scipy.stats import qmc
from itertools import product

def evaluate_discrepancy(params, n_samples=1000):
    kx, ky, kz = params
    i = np.arange(n_samples)
    
    # Vectorized generation of your polynomial sequence
    s1 = (i**1 * 1/(1+kx)) % 1
    s2 = (i**2 * 1/(1+ky)) % 1
    s3 = (i**3 * 1/(1+kz)) % 1
    sample = np.stack([s1, s2, s3], axis=1)
    
    # We evaluate the discrepancy of all 2D projections
    d12 = qmc.discrepancy(sample[:, [0, 1]])
    d13 = qmc.discrepancy(sample[:, [0, 2]])
    d23 = qmc.discrepancy(sample[:, [1, 2]])
    
    # Minimizing the 'worst-case' projection is often more stable
    return max(d12, d13, d23)

# Example search space around your initial guesses
k_space = [np.sqrt(x) for x in [2,3,5,7,11,13,17]]
best_disc = np.inf
best_ks = None

for k in product(k_space, repeat=3):
    # print(np.unique(k, return_counts=1))
    if len(np.unique(k))!=3: continue
    current_disc = evaluate_discrepancy(k)
    # print(f"k: {k}")
    # print(f"current_disc: {current_disc}")
    if current_disc < best_disc:
        best_disc = current_disc
        best_ks = k

print(f"Optimal constants: {best_ks} with Discrepancy: {best_disc}")