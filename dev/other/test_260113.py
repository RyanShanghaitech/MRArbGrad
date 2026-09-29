import math
from typing import List
from numpy import *
from matplotlib.pyplot import *

GOLDRAT = 1.6180339887498949e0  # golden ratio (φ)

def _permute(ny, nx, i) -> List[int]:
    n = ny*nx
    skip = int(n*GOLDRAT/(GOLDRAT+1))
    while math.gcd(skip, n) != 1:
        skip += 1
    
    y = (i*skip)//nx%ny
    x = (i*skip)%nx
    
    return y,x,

def _permute_improved(ny, nx, i) -> List[int]:
    n = ny*nx
    skip = 1
    lstSkip = []
    lstObj = []
    for skip in range(n):
        lstSkip.append(skip)
        lstObj.append()
    skip = lstSkip[argmin(lstObj)]
    
    y = (i*skip)//nx%ny
    x = (i*skip)%nx
    
    return y,x,

def _permute_bitrev(ny, nx, i, base=5):
    n = ny*nx
    digits = 0
    N = 1
    while N < n:
        N *= base
        digits += 1

    indices = arange(N, dtype=int)
    permuted_indices = zeros(N, dtype=int)
    
    temp_indices = indices.copy()
    
    for iDig in range(digits):
        remainder = temp_indices % base
        
        permuted_indices = permuted_indices * base + remainder
        
        temp_indices //= base
    
    if N != n:
        mask = permuted_indices < n
        permuted_indices = permuted_indices[mask]
    
    y = permuted_indices[i]//nx%ny
    x = permuted_indices[i]%nx
    
    return y,x,

def _permute_dual(ny, nx, i) -> List[int]:
    n = ny*nx
    dy0 = dy = int(ny*1/(sqrt(2)+1))
    dx0 = dx = int(nx*1/(sqrt(3)+1))
    skip = dy*nx + dx
    while math.gcd(skip, n) != 1:
        if abs(dy-dy0) > abs(dx-dx0):
            dx += 1
        else:
            dy += 1
        skip = dy*nx + dx
    print("abs(incY-incY0)", abs(dy-dy0))
    print("abs(incX-incX0)", abs(dx-dx0))
    
    y = (i*skip)//nx%ny
    x = (i*skip)%nx
    
    return y,x,

def _permute_golden_ratio(ny, nx, i): # not full random
    """
    Maps index i to (y, x) using the Golden Ratio for low-discrepancy sampling.
    This is not a strict integer permutation (it uses floats), but results 
    in superior visual distribution.
    """
    g = 1.32471795724474602596
    a1 = 1.0 / g
    a2 = 1.0 / (g * g)
    
    x_float = (0.5 + a1 * i) % 1.0
    y_float = (0.5 + a2 * i) % 1.0
    
    y = (y_float * ny).astype(int)
    x = (x_float * nx).astype(int)
    
    return y, x

def _permute_knuth(ny, nx, i): # still has stringing artifact
    n = ny * nx
    
    multiplier = int(n * (math.sqrt(5) - 1) / 2)
    while math.gcd(multiplier, n) != 1:
        multiplier += 1
        
    offset = 12345 
    
    idx_1d = (i * multiplier + offset) % n
    
    y = idx_1d // nx
    x = idx_1d % nx
    return y, x

PLASTIC_CONSTANT = 1.32471795724474602596
def _permute_r2_quasi(ny, nx, i): # the same as _permute_golden_ratio?
    
    g = PLASTIC_CONSTANT
    a1 = 1.0 / g
    a2 = 1.0 / (g * g)
    
    x_float = (0.5 + a1 * i) % 1.0
    y_float = (0.5 + a2 * i) % 1.0
    
    y = (y_float * ny).astype(int64)
    x = (x_float * nx).astype(int64)
    
    return y, x

def _permute_bit_reversal(ny, nx, i):
    n_pixels = ny * nx
    
    phi = (math.sqrt(5) - 1) / 2
    step = int(n_pixels * phi)
    
    while math.gcd(step, n_pixels) != 1:
        step += 1
        
    permuted_indices = (i * step) % n_pixels
    
    y = permuted_indices // nx
    x = permuted_indices % nx
    
    shift_factor = int(nx * phi)
    x = (x + y * shift_factor) % nx
    
    return y, x

def _permute_r2_integer(ny, nx, i):
    n = ny * nx
    
    phi = 0.6180339887498948482
    step = int(n * phi)
    if step % 2 == 0: step += 1 # Make odd
    while math.gcd(step, n) != 1:
        step += 2
        
    idx = (i * step) % n
    
    y_raw = idx // nx
    x_raw = idx % nx
    
    x_shear = (x_raw + (y_raw * 15485863)) % nx # 15485863 is the millionth prime
    
    return y_raw, x_shear

def _generate_hilbert_curve(order: int):
    y = np.array([0], dtype=np.int64)
    x = np.array([0], dtype=np.int64)
    
    for i in range(order):
        L = 1 << i # Current size
        n_curr = L * L
        
        y_0 = x
        x_0 = y
        
        y_1 = y + L
        x_1 = x
        
        y_2 = y + L
        x_2 = x + L
        
        y_3 = (L - 1) - x
        x_3 = (2 * L - 1) - y
        
        y = np.concatenate([y_0, y_1, y_2, y_3])
        x = np.concatenate([x_0, x_1, x_2, x_3])
        
    return y, x

def _permute_hilbert_golden(ny: int, nx: int, i: np.ndarray):
    max_dim = max(ny, nx)
    order = int(math.ceil(math.log2(max_dim)))
    
    hy, hx = _generate_hilbert_curve(order)
    
    valid_mask = (hy < ny) & (hx < nx)
    valid_y = hy[valid_mask]
    valid_x = hx[valid_mask]
    
    n_pixels = len(valid_y)
    
    phi = (math.sqrt(5) - 1) / 2
    step = int(n_pixels * phi)
    
    if step % 2 == 0: step += 1
    while math.gcd(step, n_pixels) != 1:
        step += 2
        
    hilbert_indices = (i * step) % n_pixels
    
    y = valid_y[hilbert_indices]
    x = valid_x[hilbert_indices]
    
    return y, x

_permute = _permute_improved

for n in range(16,512+16,16):
    print("n", n)
    arrY, arrX = _permute(n,n,arange(n*n))
    print(arrY.shape)
    print(arrX.shape)

    figure(dpi=150)
    nrow = 2
    ncol = 4
    step = n*n//(nrow*ncol)
    for iFig in range(nrow*ncol):
        subplot(nrow, ncol, 1+iFig)
        img = zeros((n,n), dtype=int64)
        s = slice(None,(iFig+1)*step)
        # s = slice(None,17)
        img[arrY[s],arrX[s]] = 1
        imshow(img)
        axis("off")
        clim(0,1)

    tight_layout(pad=0.1)
    savefig(f"example_dev/n_{n}.png", dpi=600)
    close()
# show()