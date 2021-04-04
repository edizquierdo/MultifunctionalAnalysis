import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

fig, ax = plt.subplots()

bar_heights = [4, 5, 6, 3, 7, 5]
save_filename = "./test.pdf"
oc_approach = [r, g, b, 1]
oc_avoid = [r, g, b, 1]
pa_approach = [r, g, b, 1]
pa_avoid = [r, g, b, 1]

color_maps = []

# OC
_cm_list = []
_cm_list += [oc_approach + [i] for i in np.linspace(1, 0, 128)]
_cm_list += [oc_avoid + [i] for i in np.linspace(0, 1, 128)]
cm = ListedColormap(_cm_list)
color_maps.append(cm)

# PA
_cm_list = []
_cm_list += [pa_approach + [i] for i in np.linspace(1, 0, 128)]
_cm_list += [pa_avoid + [i] for i in np.linspace(0, 1, 128)]
cm = ListedColormap(_cm_list)
color_maps.append(cm)

# Avoid
_cm_list = []
_cm_list += [pa_avoid + [i] for i in np.linspace(1, 0, 128)]
_cm_list += [oc_avoid + [i] for i in np.linspace(0, 1, 128)]
cm = ListedColormap(_cm_list)
color_maps.append(cm)

# Approach
_cm_list = []
_cm_list += [pa_approach + [i] for i in np.linspace(1, 0, 128)]
_cm_list += [oc_approach + [i] for i in np.linspace(0, 1, 128)]
cm = ListedColormap(_cm_list)
color_maps.append(cm)


plt.figure(figsize=[4, 3])
bar = ax.bar(np.arange(1, len(bar_heights) + 1), bar_heights)


def gradientbars(bars):
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


gradientbars(bar)
plt.savefig(save_filename)
plt.show()
