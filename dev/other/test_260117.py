import numpy as np
from matplotlib.pyplot import *
from itertools import product

def _rsi_permutation(shape, seed=0):
    """
    Recursive Stratified Interleaving (RSI).
    Returns a permutation of all grid points for arbitrary dimension.

    Good for progressive, uniform prefixes.
    """
    rng = np.random.default_rng(seed)
    d = len(shape)

    def rec(ranges):
        # ranges: list of (lo, hi) half-open per dimension
        sizes = [hi - lo for (lo, hi) in ranges]
        if all(s == 1 for s in sizes):
            return [tuple(lo for (lo, hi) in ranges)]

        # split each dimension at midpoint
        mids = [(lo + hi) // 2 for (lo, hi) in ranges]
        subranges = []

        for bits in product([0, 1], repeat=d):
            sr = []
            valid = True
            for dim, b in enumerate(bits):
                lo, hi = ranges[dim]
                mid = mids[dim]
                if b == 0:
                    lo2, hi2 = lo, mid
                else:
                    lo2, hi2 = mid, hi
                if lo2 >= hi2:
                    valid = False
                    break
                sr.append((lo2, hi2))
            if valid:
                subranges.append(sr)

        # recurse
        sequences = [rec(sr) for sr in subranges]
        # rng.shuffle(sequences)

        # interleave round-robin
        out = []
        maxlen = max(len(seq) for seq in sequences)
        for k in range(maxlen):
            for seq in sequences:
                if k < len(seq):
                    out.append(seq[k])
        return out

    ranges = [(0, s) for s in shape]
    coords = rec(ranges)
    return np.array(coords, dtype=np.int64)

def _permute_rsi(ny, nx, seed=0):
    coords = _rsi_permutation((ny, nx), seed=seed)
    arrY = coords[:, 0]
    arrX = coords[:, 1]
    return arrY, arrX

# ---- Visualization (mirrors your loop) ----
for n in range(16, 512 + 16, 16):
    print("n", n)
    arrY, arrX = _permute_rsi(n, n, seed=42)
    print(arrY.shape)
    print(arrX.shape)

    figure(dpi=150)
    nrow = 2
    ncol = 4
    step = n * n // (nrow * ncol)
    for iFig in range(nrow * ncol):
        subplot(nrow, ncol, 1 + iFig)
        img = np.zeros((n, n), dtype=np.int64)
        s = slice(None, (iFig + 1) * step)
        img[arrY[s], arrX[s]] = 1
        imshow(img)
        axis("off")
        clim(0, 1)

    tight_layout(pad=0.1)
    savefig(f"example_dev/n_{n}.png", dpi=600)
    close()