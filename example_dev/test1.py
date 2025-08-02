from numpy import *
from matplotlib.pyplot import *

# Constants
alpha_deg = 10
alpha = deg2rad(alpha_deg)
cos_alpha = cos(alpha)
sin_alpha = sin(alpha)

# Tissue parameters
t1_gm = 1286    # ms
t2s_gm = 117     # ms
pd_gm = 0.86

t1_wm = 788    # ms
t2s_wm = 80.4     # ms
pd_wm = 0.73

def spgr_signal(fa, tr, te, t1, t2star, pd):
    e1 = exp(-tr / t1)
    return pd * sin(fa) * (1 - e1) / (1 - cos(fa) * e1) * exp(-te / t2star)

best_fa = 0
best_tr = 0
max_contrast = 0

te = 1
tr_values = linspace(2, 30, 100)  # TE: 0.1 to 30 ms (step 0.1 ms)
fa_values = linspace(1, 90, 100)*pi/180

mesh_contrast = zeros([tr_values.size, fa_values.size], dtype=float64)
mesh_signal = zeros([tr_values.size, fa_values.size], dtype=float64)

for i in range(tr_values.size):
    tr = tr_values[i]
    for j in range(fa_values.size):
        fa = fa_values[j]
        s_gm = spgr_signal(fa, tr, te, t1_gm, t2s_gm, pd_gm)
        s_wm = spgr_signal(fa, tr, te, t1_wm, t2s_wm, pd_wm)
        
        contrast = s_wm/s_gm
        if contrast>=1: contrast -= 1
        elif contrast<1: contrast = 1 - 1/contrast
        mesh_contrast[i,j] = contrast
        mesh_signal[i,j] = max(s_gm, s_wm)
        if abs(contrast) > max_contrast:
            max_contrast = abs(contrast)
            best_fa = fa
            best_tr = tr

print(f"Best FA: {best_fa*180/pi:.1f} deg, Best TR: {best_tr:.1f} ms, Max S_WM/S_GM: {max_contrast:.4f}")

fig = figure(figsize=(10,5))

ax = subplot(121)
imax = ax.imshow(mesh_contrast, cmap="coolwarm", vmin=-0.5, vmax=+0.5)
colorbar(imax)
ax.set_yticks(arange(tr_values.size)[::20], char.mod('%.1f', tr_values[::20]))
ylabel("TR")
ax.set_title("WM / GM")
ax.set_xticks(arange(fa_values.size)[::20], char.mod('%.1f', fa_values[::20]*180/pi), rotation=90)
xlabel("FA")

ax = subplot(122)
imax = ax.imshow(mesh_signal)
colorbar(imax)
ax.set_yticks(arange(tr_values.size)[::20], char.mod('%.1f', tr_values[::20]))
ylabel("TR")
ax.set_title("WM / GM")
ax.set_xticks(arange(fa_values.size)[::20], char.mod('%.1f', fa_values[::20]*180/pi), rotation=90)
xlabel("FA")

figure()
ax = subplot(111)
imax = imshow(mesh_contrast*mesh_signal, cmap="coolwarm", vmin=-0.1, vmax=+0.1)
colorbar(imax)
ax.set_yticks(arange(tr_values.size)[::20], char.mod('%.1f', tr_values[::20]))
ylabel("TR")
ax.set_title("WM / GM")
ax.set_xticks(arange(fa_values.size)[::20], char.mod('%.1f', fa_values[::20]*180/pi), rotation=90)
xlabel("FA")

show()