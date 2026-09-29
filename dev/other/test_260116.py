from numpy import *
from matplotlib.pyplot import *

def _permute(n, base=2):
    digits = 0
    N = 1
    while N < n:
        N *= base
        digits += 1

    indices = arange(N, dtype=int)
    permuted_indices = zeros(N, dtype=int)
    
    temp_indices = indices.copy()
    
    for i in range(digits):
        remainder = temp_indices % base
        
        permuted_indices = permuted_indices * base + remainder
        
        temp_indices //= base
    
    if N != n:
        mask = permuted_indices < n
        permuted_indices = permuted_indices[mask]

    return permuted_indices

# --- Test Code ---
n0 = 32; nSamp = 4
p0 = _permute(n0, base=4)[:nSamp]
# p1 = _permute(n0, base=2)[:nSamp]
img = zeros((n0,n0), dtype=bool)


fig = figure()
ax = subplot(111)
imax = imshow(img)
imax.set_clim(0,1)
for i in p0:
    # for j in p1:
    # img[i,j] = 1
    img.flat[i] = 1
    imax.set_data(img)
    draw()
    pause(1e-0)

show()
exit()

# Test Case 1: Power of two (n=8)
n1 = 128
p1 = _permute(n1)
print(f"Bit-reversal permutation for n={n1}:")
print(p1)

n2 = 111
p2 = _permute(n2)
print(f"\nBit-reversal permutation for n={n2}:")
print(p2)

# Visualization
figure(figsize=(10, 4))

subplot(1, 2, 1)
title(f"Permutation Map (n={n1})")
plot(arange(n1), p1, 'o-', label='Permutation')
xlabel("Original Index")
ylabel("Permuted Index")
grid(True)

subplot(1, 2, 2)
title(f"Permutation Map (n={n2})")
plot(arange(n2), p2, 'o-', color='orange', label='Permutation')
xlabel("Original Index")
ylabel("Permuted Index")
grid(True)

tight_layout()
show()