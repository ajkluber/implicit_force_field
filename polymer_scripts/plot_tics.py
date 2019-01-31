from __future__ import print_function
import os
import sys
import glob
import argparse
import numpy as np 

from scipy.stats import binned_statistic_2d as bin2d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

def data_filename(dataname, idx1, idx2):
    return "run_{}/{}_{}.npy".format(idx1, dataname, idx2)

def plot_tic_vs_tic(tics, msm_savedir):

    #fig, axes = plt.subplots(5, 1, sharex=True)
    #for i in range(5):
    #    ax = axes[i]
    #    ax.plot(tics[i][0][:10000])
    #    ax.set_ylabel("TIC " + str(i + 1))
    #fig.savefig("tic_subplot.pdf")
    #fig.savefig("tic_subplot.png")

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

            pcol = ax.pcolormesh(X, Y, Hmsk, linewidth=0, rasterized=True)
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

    axes[0][0].annotate("TICA  " + msm_savedir.split("msm_")[1], fontsize=24, xy=(0,0),
            xytext=(1.8, 1.1), xycoords="axes fraction", textcoords="axes fraction")
    fig.savefig(msm_savedir + "/tic_hist_grid.pdf")
    fig.savefig(msm_savedir + "/tic_hist_grid.png")

def plot_rg_vs_tics(traj_idxs, tics, msm_savedir):
    print("Plotting tics vs radius of gyration...")
    all_rg = []
    for n in range(len(traj_idxs)):
        idx1, idx2 = traj_idxs[n]
        all_rg.append(np.load(data_filename("rg", idx1, idx2)))

    fig, axes = plt.subplots(1, 4, figsize=(20,5))
    for i in range(4):
        ax = axes[i]
        x = np.concatenate(tics[i])
        y = np.concatenate(all_rg)

        #import pdb 
        #pdb.set_trace()
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

    axes[1].annotate("TIC  " + msm_savedir.split("msm_")[1], fontsize=24, xy=(0,0),
            xytext=(1., 1.1), xycoords="axes fraction", textcoords="axes fraction")
    fig.savefig(msm_savedir + "/tic_vs_rg_hist.pdf")
    fig.savefig(msm_savedir + "/tic_vs_rg_hist.png")

def plot_gyr_moments_vs_tics(traj_idxs, tics, msm_savedir):
    print("Plotting tics vs gyration moments...")

    # sort moments of 
    all_gyr = [[],[], []]
    for n in range(len(traj_idxs)):
        idx1, idx2 = traj_idxs[n]
        gyr_temp = np.load(data_filename("gyr_moments", idx1, idx2))
        gyr_temp.sort()
        all_gyr[0].append(gyr_temp[:,2])
        all_gyr[1].append(gyr_temp[:,1])
        all_gyr[2].append(gyr_temp[:,0])


    gyr_ylim = [(2, 9), (0.5, 4), (0.25, 2.25)]

    # TIC vs gyration moments
    fig, axes = plt.subplots(3, 4, figsize=(20,20))
    for i in range(4):
        for j in range(3):
            ax = axes[j, i]
            x = np.concatenate(tics[i])
            y = 2*np.sqrt(np.concatenate(all_gyr[j]))
            corr_coeff = np.corrcoef(x,y)[0,1]

            H, xedges, yedges = np.histogram2d(x, y, bins=100)
            X, Y = np.meshgrid(xedges, yedges)
            Hmsk = np.ma.array(H, mask=H ==0)

            pcol = ax.pcolormesh(X, Y, Hmsk.T, linewidth=0, rasterized=True)
            pcol.set_edgecolor("face")
                
            #ax.hist2d(x, y, bins=100)
            if i == 0:
                ax.set_ylabel(r"$\xi_" + str(j+1) + r"$ ($\AA$)", fontsize=26)
            if j == 2:
                #ax.set_xlabel(r"$\psi_" + str(i + 1) + "$", fontsize=26)
                ax.set_xlabel(r"TIC " + str(i + 1), fontsize=26)
            ax.set_ylim(*gyr_ylim[j])

            ax.annotate(r"Corr$= {:.3f}$".format(corr_coeff), fontsize=20, xy=(0,0),
                xytext=(0.4, .92), xycoords="axes fraction", textcoords="axes fraction",
                bbox=dict(fc="white", ec="k", lw=2))

    fig.savefig(msm_savedir + "/tic_vs_gyr_hist.pdf")
    fig.savefig(msm_savedir + "/tic_vs_gyr_hist.png")

    # plot histogram of tica coordinates
    for n in range(3):
        gyr = 2*np.sqrt(np.concatenate(all_gyr[n]))

        fig, axes = plt.subplots(4, 4, figsize=(20,20))
        for i in range(4):
            for j in range(i, 4):
                ax = axes[i][j]
                x = np.concatenate(tics[i])
                y = np.concatenate(tics[j + 1])

                H, xedges, yedges, _ = bin2d(x, y, gyr, statistic="mean", bins=100)

                X, Y = np.meshgrid(xedges, yedges)
                Hmsk = np.ma.array(H, mask=H ==0)
                #ax.hist2d(np.concatenate(tics[i]), , bins=100)

                pcol = ax.pcolormesh(X, Y, Hmsk, linewidth=0, rasterized=True)
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

        #axes[0][0].annotate("TICA  " + msm_savedir.split("msm_")[1], fontsize=24, xy=(0,0),
        #        xytext=(1.8, 1.1), xycoords="axes fraction", textcoords="axes fraction")
        fig.savefig(msm_savedir + "/tic_hist_color_by_gyr_moment_{}.pdf".format(n+1))
        fig.savefig(msm_savedir + "/tic_hist_color_by_gyr_moment_{}.png".format(n+1))

def plot_helix_vs_tics(traj_idxs, tics, msm_savedir):
    all_L = []
    all_R = []
    for n in range(len(traj_idxs)):
        idx1, idx2 = traj_idxs[n]
        all_L.append(np.load("run_{}/helix_L_{}.npy".format(idx1, idx2)))
        all_R.append(np.load("run_{}/helix_R_{}.npy".format(idx1, idx2)))

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

    axes[0,1].annotate("TIC  " + msm_savedir.split("msm_")[1], fontsize=24, xy=(0,0),
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

    #axes[0][0].annotate("TICA  " + msm_savedir.split("msm_")[1], fontsize=24, xy=(0,0),
    #        xytext=(1.8, 1.1), xycoords="axes fraction", textcoords="axes fraction")
    fig.savefig("helix_hist.pdf")
    fig.savefig("helix_hist.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("msm_savedir", type=str)
    parser.add_argument("--keep_dims", type=int, default=5)
    parser.add_argument("--plot_tics", action="store_true")
    parser.add_argument("--plot_rg", action="store_true")
    parser.add_argument("--plot_helix", action="store_true")
    parser.add_argument("--plot_gyr", action="store_true")
    args = parser.parse_args()

    name = args.name
    msm_savedir = args.msm_savedir
    keep_dims = args.keep_dims
    plot_tics = args.plot_tics 
    plot_rg = args.plot_rg   
    plot_gyr = args.plot_gyr   
    plot_helix = args.plot_helix

    #python ~/code/implicit_force_field/polymer_scripts/plot_tics.py c25 msm_dists --plot_rg

    # What about runs with multiple trajectories?
    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd") 
    traj_idxs = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        traj_idxs.append([idx1, idx2])

    tics = [] 
    for i in range(keep_dims):
        temp_tic = []
        for n in range(len(traj_idxs)):
            idx1, idx2 = traj_idxs[n]

            # save TIC with indices of corresponding traj
            tic_saveas = msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, i+1)
            temp_tic.append(np.load(tic_saveas))
        tics.append(temp_tic)

    if plot_tics:
        plot_tic_vs_tic(tics, msm_savedir)

    rg_files_exist = np.all([ os.path.exists(data_filename("rg", idx1, idx2)) for idx1, idx2 in traj_idxs ])
    if plot_rg and rg_files_exist:
        plot_rg_vs_tics(traj_idxs, tics, msm_savedir)

    gyr_files_exist = np.all([ os.path.exists(data_filename("gyr_moments", idx1, idx2)) for idx1, idx2 in traj_idxs ])
    if plot_gyr and gyr_files_exist: 
        plot_gyr_moments_vs_tics(traj_idxs, tics, msm_savedir)

    helix_files_exist = np.all([ os.path.exists(data_filename("helix_L_", idx1, idx2)) for idx1, idx2 in traj_idxs ])
    if plot_helix and helix_files_exist:
        plot_helix_vs_tics(traj_idxs, tics, msm_savedir)

