import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap


def make_cmap(from_color, to_color):
    from_color = np.array(from_color)
    to_color = np.array(to_color)
    _cm_list = [(to_color * i + from_color * (1 - i)) for i in np.linspace(0, 1, 256)]
    cm = ListedColormap(_cm_list)
    return cm


def gradientbars(bars, color_maps):
    grad = np.atleast_2d(np.linspace(0, 1, 256)).T
    ax = bars[0].axes
    lim = ax.get_xlim() + ax.get_ylim()
    for bar, cm in zip(bars, color_maps):
        bar.set_zorder(1)
        bar.set_facecolor("none")
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto", zorder=0, cmap=cm)
    ax.axis(lim)


FC_Pearson = [0.127704, 0.06499873157680602, 0.171356, 0.07095927879284398]
FC_NA = [1.29903551955000642, 1.51367327519719448, 0.735576115263650184, 1.7213432648094396]
FC_MI = [1.12556, 0.6416288822013887, 0.940723, 0.8432075108869090]
N_VAR = [0.295563, 0.297003, 0.300797, 0.293771]
N_MI = [0.203838252678301902, 0.188140514752626004, 0.181514443232673108, 0.18092136629698959]

bar_heights = N_MI
save_filename = "./N_MI.pdf"
oc_approach = [214.0 / 255, 158.0 / 255, 64.0 / 255]
oc_avoid = [101.0 / 255, 128.0 / 255, 177.0 / 255]
pa_avoid = [150.0 / 255, 175.0 / 255, 72.0 / 255]
pa_approach = [201.0 / 255, 104.0 / 255, 95.0 / 255]

color_maps = []

# OC
color_maps.append(make_cmap(oc_avoid, oc_approach))

# PA
color_maps.append(make_cmap(pa_avoid, pa_approach))

# Avoid
color_maps.append(make_cmap(oc_avoid, pa_avoid))

# Approach
color_maps.append(make_cmap(oc_approach, pa_approach))


plt.figure(figsize=[4, 2])
ax = plt.gca()
bar_plt = ax.bar(np.arange(1, len(bar_heights) + 1), bar_heights)
gradientbars(bar_plt, color_maps)

plt.xticks([1, 2, 3, 4], ["OC", "PA", "Aviod", "Approach"])
plt.ylabel("Euclidean distance")

plt.tight_layout()
plt.savefig(save_filename)
plt.show()
