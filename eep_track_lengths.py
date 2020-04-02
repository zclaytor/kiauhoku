import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import kiauhoku as kh


def my_hist(data, title, eep_indices):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fs = 14
    fontdict = {"fontsize": fs}
    labeldict = {"labelsize": fs}

    ax.imshow(data, aspect="auto", cmap=my_cmap(), norm=my_norm(eep_indices), 
              extent=(0.295, 2.005, 0.75, -1.25), picker=True)
    ax.invert_yaxis()
    ax.set_title(title, **fontdict)
    ax.set_ylabel("[M/H]", **fontdict)
    ax.set_xlabel(r"$M$ / $M_\odot$", **fontdict)
    ax.set_xticks(np.linspace(0.305, 1.995, 170), minor=True)
    ax.set_xticks(np.linspace(0.3, 2.0, 18), minor=False)
    ax.set_yticks(np.linspace(-1.25, 0.75, 5), minor=True)
    ax.set_yticks(np.linspace(-1.0, 0.5, 4), minor=False)
    ax.xaxis.grid(True, which="minor", color="gray", linestyle="--")
    ax.yaxis.grid(True, which="minor", color="k")
    ax.tick_params(axis="both", which="minor", length=0)
    ax.tick_params(axis="both", which="major", **labeldict)

    textdict = {"rotation": 90, 
                "fontsize": 18,
                "horizontalalignment": "center",
                "verticalalignment": "center"}

    ax.text(0.8, -1.0, "RGBump", **textdict)
    ax.text(0.66,-1.0, "TAMS", **textdict)
    ax.arrow(0.63, -1.0, -0.02, 0, zorder=5, 
             width=0.002, head_width=0.05, head_length=0.02, fc="k")
    ax.text(0.53, -1.0, "IAMS", **textdict)
    ax.text(0.40, -1.0, "EAMS", **textdict)
    ax.text(0.32, -1.0, "ZAMS", **textdict)

    return fig, ax


def my_cmap():
    clist = ["black", "salmon", "sandybrown", "khaki", "mediumseagreen", "skyblue"]
    cmap = mpl.colors.ListedColormap(clist)
    return cmap

def my_norm(eep_indices):
    bounds = [-1] + [i+2 for i in eep_indices]
    norm = mpl.colors.BoundaryNorm(bounds, my_cmap().N)
    return norm


def length_distribution(grid0, grid1, verbose=True):
    lens = []
    n_len = []
    for i in grid0:
        for j in i:
            if j not in lens:
                lens.append(j)
                n_len.append(1)
            else:
                idx = lens.index(j)
                n_len[idx] += 1
    for i in grid1:
        for j in i:
            if j not in lens:
                lens.append(j)
                n_len.append(1)
            else:
                idx = lens.index(j)
                n_len[idx] += 1

    if verbose:
        print("Lengths:                ", lens)
        print("Number of each length:  ", n_len)
        print("Total number of tracks: ", sum(n_len))

    return sorted(lens)


def measure_lengths(grid_name):
    eep_grid = kh.load_eep_grid(grid_name)
    eep_lengths = eep_grid.get_eep_track_lengths()

    metallicities = [-1.0, -0.5, 0.0, 0.5]
    masses = np.linspace(30, 200, 171)/100
    grid0 = np.zeros((4, 171), dtype=int)
    grid1 = np.zeros((4, 171), dtype=int)

    for i, z in enumerate(metallicities):
        for j, m in enumerate(masses):
            try:
                grid0[i, j] = eep_lengths.loc[(m, z, 0), 0]
            except:
                pass
            try:
                grid1[i, j] = eep_lengths.loc[(m, z, 0.4), 0]
            except:
                pass
    
    lens = length_distribution(grid0, grid1)
    return grid0, grid1


def make_plots(grid0, grid1, eep_indices):
    f1, a1 = my_hist(grid0, r"[$\alpha$/M] = 0.0", eep_indices)
    f2, a2 = my_hist(grid1, r"[$\alpha$/M] = 0.4", eep_indices)
    #f1.canvas.mpl_connect("button_press_event", onclick000)
    #f2.canvas.mpl_connect("button_press_event", onclick040)
    xlim = a1.get_xlim()
    ylim = a1.get_ylim()

    #a1.plot(data["mass"], data["met"], "ko", ms=3)
    #a2.plot(data["mass"], data["met"], "ko", ms=3)
    a1.set_xlim(xlim)
    a1.set_ylim(ylim)
    a2.set_xlim(xlim)
    a2.set_ylim(ylim)

    plt.show()

'''
def onclick000(event):
    try:
        mass = round(event.xdata, 2)
        met = round(2*(event.ydata))/2
        print("Mass: %.2f Msun | [M/H]: %.1f" %(mass, met))
        alpha = 0
        _HRD_model(mass, met, alpha)
    except TypeError:
        pass


def onclick040(event):
    try:
        mass = round(event.xdata, 2)
        met = round(2*(event.ydata))/2
        print("Mass: %.2f Msun | [M/H]: %.1f" %(mass, met))
        alpha = 0.4
        _HRD_model(mass, met, alpha)
    except TypeError:
        pass
'''

if __name__ == "__main__":
    name = 'fastlaunch'
    eep_params = kh.stargrid.load_eep_params(name)
    intervals = eep_params['intervals']
    eep_indices = np.arange(len(intervals) + 1)
    eep_indices[1:] += np.cumsum(intervals)

    grid0, grid1 = measure_lengths(name)
    make_plots(grid0, grid1, eep_indices)
