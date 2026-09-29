from numpy import *
import numpy as np

import random
from random import seed as py_seed

from numpy.linalg import norm
import mrautograd as mag

from scipy.io import savemat, loadmat

# ---------------- constants ----------------
gamma = 42.5756e6
fov = 0.256
nPix = 256
sLim = 100 * gamma * fov / nPix
gLim = 120e-3 * gamma * fov / nPix
dtGrad = 10e-6
dtADC = dtGrad

nAx = 2
nCity = 256**nAx // 256

# scale used in your print/plots (convert to "mT/m, T/m/s-like" display units you used)
disp_scale = (nPix / fov) / (42.58e6)

# ---------------- helpers ----------------
def compute_smax_for_seed(s, use_cache=True, cache_path="/mnt/d/LProject/minTimeGradient/Matlab/arrCity.mat"):
    # Ensure both Python's and NumPy's RNG are seeded
    py_seed(s)
    random.seed(s)
    np.random.seed(s)

    # Generate/load cities for this seed
    if use_cache:
        # If you cache, note: caching makes ALL seeds reuse the same arrCity (so sMax won't change).
        # Use cache only if that's what you want.
        arrCity = loadmat(cache_path)["arrCity"]
    else:
        arrCity = mag.Utility.genTspTraj(nCity)
        arrCity = mag.Utility.rmCity(arrCity, pi/6, 8/nPix)

    # Solver config
    mag.setSolverMtg(0)

    # Slew-rate constrained gradient
    arrG, _ = mag.calGrad4ExSamp(False, fov, nPix, sLim, gLim, dtGrad, arrCity)

    # Slew
    arrS = diff(arrG, axis=0) / dtGrad

    # sMax in your displayed units
    smax_disp = np.max(norm(arrS, axis=-1)) * disp_scale
    return float(smax_disp)

# ---------------- sweep seeds ----------------
use_cache = False  # set True only if you *want* same arrCity for all seeds
results = []
for s in range(100):
    try:
        smax = compute_smax_for_seed(s, use_cache=use_cache)
        results.append((s, smax))
        print(f"seed={s:02d}  sMax={smax:.6f}")
    except Exception as e:
        results.append((s, np.nan))
        print(f"seed={s:02d}  FAILED: {e}")

# Optional: save to MAT/NPY/CSV if you want
# from scipy.io import savemat
# savemat("sweep_smax.mat", {"seed": np.array([r[0] for r in results]),
#                            "sMax": np.array([r[1] for r in results])})

# Quick summary
valid = [(s, v) for s, v in results if np.isfinite(v)]
if valid:
    best = min(valid, key=lambda x: x[1])
    worst = max(valid, key=lambda x: x[1])
    print("\nSummary:")
    print(f"min sMax: seed={best[0]}  sMax={best[1]:.6f}")
    print(f"max sMax: seed={worst[0]}  sMax={worst[1]:.6f}")
else:
    print("\nNo valid results.")