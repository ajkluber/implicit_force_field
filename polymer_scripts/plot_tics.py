import os
import sys
import glob
import argparse
import numpy as np 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep_dims", type=int, default=5)
    parser.add_argument("--use_dihedrals", action="store_true")
    parser.add_argument("--use_distances", action="store_true")
    parser.add_argument("--use_inv_distances", action="store_true")
    parser.add_argument("--use_rg", action="store_true")
    args = parser.parse_args()

    keep_dims = args.keep_dims
    use_dihedrals = args.use_dihedrals
    use_distances = args.use_distances
    use_inv_distances = args.use_inv_distances
    use_rg = args.use_rg

    #python ~/code/implicit_force_field/polymer_scripts/plot_tics.py --use_dihedrals --use_distances --keep_dims 5

    # determine input features
    feature_set = []
    if use_dihedrals:
        feature_set.append("dih")
    if use_distances:
        feature_set.append("dists")
    if use_inv_distances:
        feature_set.append("invdists")
    if use_rg:
        feature_set.append("rg")

    f_str = "_".join(feature_set)
    msm_savedir = "msm_" + f_str

    os.chdir(msm_savedir)
    n_runs = len(glob.glob("run_*_TIC_1.npy"))

    # What about runs with multiple trajectories?

    tics = []
    for i in range(keep_dims):
        tic_i = []
        #tic_filenames = glob.glob("run_*_TIC_{}.npy".format(i+1))
        for n in range(n_runs):
            tic_i.append(np.load("run_{}_TIC_{}.npy".format(n+1,i+1)))
        tics.append(tic_i)

    all_rg = []
    for n in range(n_runs):
        all_rg.append(np.load("../run_{}/rg_1.npy".format(n+1)))

    all_L = []
    all_R = []
    for n in range(n_runs):
        all_L.append(np.load("../run_{}/helix_L_1.npy".format(n+1)))
        all_R.append(np.load("../run_{}/helix_R_1.npy".format(n+1)))


    fig, axes = plt.subplots(5, 1, sharex=True)
    for i in range(5):
        ax = axes[i]
        ax.plot(tics[i][0][:10000])
        ax.set_ylabel("TIC " + str(i + 1))
    fig.savefig("tic_subplot.pdf")
    fig.savefig("tic_subplot.png")

    # plot histogram of tica coordinates
    fig, axes = plt.subplots(4, 4, figsize=(20,20))
    for i in range(4):
        for j in range(i, 4):
            ax = axes[i][j]
            x = np.concatenate(tics[i])
            y = np.concatenate(tics[j + 1])

            H, xedges, yedges = np.histogram2d(x, y, bins=100)
            X, Y = np.meshgrid(xedges, yedges)
            Hmsk = np.ma.array(H, mask=H ==0)
            #ax.hist2d(np.concatenate(tics[i]), , bins=100)

            pcol = ax.pcolormesh(X, Y, Hmsk.T, linewidth=0, rasterized=True)
            pcol.set_edgecolor("face")

            if i == 3:
                ax.set_xlabel("TIC " + str(i + 2), fontsize=20)
            #if j == 0:
            #    ax.set_ylabel("TIC " + str(j + 1), fontsize=20)
            #    #ax.set_title("TIC " + str(i + 2), fontsize=20)

        axes[i][0].set_ylabel("TIC " + str(i + 1), fontsize=20)

        if i == 3:
            for j in range(4):
                axes[i][j].set_xlabel("TIC " + str(j + 2), fontsize=20)

    axes[0][0].annotate("TICA  " + f_str, fontsize=24, xy=(0,0),
            xytext=(1.8, 1.1), xycoords="axes fraction", textcoords="axes fraction")
    fig.savefig("tic_hist_grid.pdf")
    fig.savefig("tic_hist_grid.png")

    fig, axes = plt.subplots(1, 4, figsize=(20,5))
    for i in range(4):
        ax = axes[i]
        x = np.concatenate(tics[i])
        y = np.concatenate(all_rg)
        corr_coeff = np.corrcoef(x,y)[0,1]

        H, xedges, yedges = np.histogram2d(x, y, bins=100)
        X, Y = np.meshgrid(xedges, yedges)
        Hmsk = np.ma.array(H, mask=H ==0)

        pcol = ax.pcolormesh(X, Y, Hmsk.T, linewidth=0, rasterized=True)
        pcol.set_edgecolor("face")
            
        #ax.hist2d(x, y, bins=100)
        if i == 0:
            ax.set_ylabel("$R_g$ (nm)", fontsize=20)
        ax.set_xlabel("TIC " + str(i + 1), fontsize=20)

        ax.annotate(r"Corr$= {:.3f}$".format(corr_coeff), fontsize=14, xy=(0,0),
            xytext=(0.5, .85), xycoords="axes fraction", textcoords="axes fraction",
            bbox=dict(fc="white", ec="k", lw=2))

    axes[1].annotate("TIC  " + f_str, fontsize=24, xy=(0,0),
            xytext=(1., 1.1), xycoords="axes fraction", textcoords="axes fraction")
    fig.savefig("tic_vs_rg_hist.pdf")
    fig.savefig("tic_vs_rg_hist.png")



    # plot TICs versus helical measure
    fig, axes = plt.subplots(2, 4, figsize=(20,10))
    for i in range(4):
        ax1 = axes[0,i]
        ax2 = axes[1,i]
        x = np.concatenate(tics[i])
        y1 = np.concatenate(all_L)
        y2 = np.concatenate(all_R)
        corr1 = np.corrcoef(x,y1)[0,1]
        corr2 = np.corrcoef(x,y2)[0,1]

        H1, xedges, yedges = np.histogram2d(x, y1, bins=100)
        X1, Y1 = np.meshgrid(xedges, yedges)
        Hmsk1 = np.ma.array(H1, mask=H1 == 0)

        H2, xedges, yedges = np.histogram2d(x, y2, bins=100)
        X2, Y2 = np.meshgrid(xedges, yedges)
        Hmsk2 = np.ma.array(H2, mask=H2 == 0)

        pcol1 = ax1.pcolormesh(X1, Y1, Hmsk1.T, linewidth=0, rasterized=True)
        pcol1.set_edgecolor("face")

        pcol2 = ax2.pcolormesh(X2, Y2, Hmsk2.T, linewidth=0, rasterized=True)
        pcol2.set_edgecolor("face")
            
        if i == 0:
            ax1.set_ylabel(r"$h_L$", fontsize=20)
            ax2.set_ylabel(r"$h_R$", fontsize=20)
        ax2.set_xlabel("TIC " + str(i + 1), fontsize=20)

        ax1.annotate(r"Corr$= {:.3f}$".format(corr1), fontsize=14, xy=(0,0),
            xytext=(0.5, .85), xycoords="axes fraction", textcoords="axes fraction",
            bbox=dict(fc="white", ec="k", lw=2))
        ax2.annotate(r"Corr$= {:.3f}$".format(corr2), fontsize=14, xy=(0,0),
            xytext=(0.5, .85), xycoords="axes fraction", textcoords="axes fraction",
            bbox=dict(fc="white", ec="k", lw=2))

    axes[0,1].annotate("TIC  " + f_str, fontsize=24, xy=(0,0),
            xytext=(1., 1.1), xycoords="axes fraction", textcoords="axes fraction")
    fig.savefig("tic_vs_helix_hist.pdf")
    fig.savefig("tic_vs_helix_hist.png")


    fig, axes = plt.subplots(1, 1, figsize=(5,5))
    ax = axes
    x = np.concatenate(all_L)
    y = np.concatenate(all_R)

    H, xedges, yedges = np.histogram2d(x, y, bins=100)
    X, Y = np.meshgrid(xedges, yedges)
    Hmsk = np.ma.array(H, mask=H ==0)

    pcol = ax.pcolormesh(X, Y, Hmsk.T, linewidth=0, rasterized=True)
    pcol.set_edgecolor("face")

    ax.set_xlabel("$h_L$", fontsize=20)
    ax.set_ylabel("$h_R$", fontsize=20)

    #axes[0][0].annotate("TICA  " + f_str, fontsize=24, xy=(0,0),
    #        xytext=(1.8, 1.1), xycoords="axes fraction", textcoords="axes fraction")
    fig.savefig("helix_hist.pdf")
    fig.savefig("helix_hist.png")
