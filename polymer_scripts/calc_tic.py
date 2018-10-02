import os
import sys
import glob
import argparse
import numpy as np 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyemma.plots as mplt

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

import pyemma.coordinates as coor
import pyemma.msm as msm
from pyemma.coordinates.data.featurization.misc import CustomFeature


import mdtraj as md

def tanh_contact(traj, pairs, r0, widths):
    r = md.compute_distances(traj, pairs)
    return 0.5*(np.tanh((r0 - r)/widths) + 1)

def rg_feature(traj):
    return md.compute_rg(traj).astype(np.float32).reshape(-1, 1)

def q_feature(traj, pairs, r0, widths):
    r = md.compute_distances(traj, pairs)
    return np.sum(0.5*(np.tanh((r0 - r)/widths) + 1), axis=1)

def local_density_feature(traj, pairs, r0, widths):

    rho_i = np.zeros((traj.n_frames, traj.n_atoms), np.float32)
    for i in range(len(pairs)):
        r = md.compute_distances(traj, pairs[i])
        rho_i[:,i] = np.sum(0.5*(np.tanh((r0 - r)/widths) + 1), axis=1)
    #r = md.compute_distances(traj, pairs)
    #rho_i = np.sum(0.5*(np.tanh((r0 - r)/widths) + 1), axis=1).astype(np.float32).reshape(-1,1)
    return rho_i.astype(np.float32)

def plot_tica_stuff():
    # calculate TICA at different lagtimes
    #tica_lags = np.array(range(1, 11) + [12, 15, 20, 25, 50, 75, 100, 150, 200])
    tica_lags = np.array([1, 5, 10, 25, 50, 100, 200, 500, 1000])
    all_cumvar = []
    all_tica_ti = []
    for i in range(len(tica_lags)):
        tica = coor.tica(lag=tica_lags[i], stride=1)
        coor.pipeline([reader, tica])

        all_cumvar.append(tica.cumvar)
        all_tica_ti.append(tica.timescales)

    all_cumvar = np.array(all_cumvar)
    all_tica_ti = np.array(all_tica_ti)

    # times vs lag
    plt.figure()
    for i in range(20):
        plt.plot(tica_lags, all_tica_ti[:,i])
    plt.fill_between(tica_lags, tica_lags, color='gray', lw=2)
    #ymin, ymax = plt.ylim()
    #plt.ylim(ymin, ymax)
    plt.grid(True, alpha=1, color='k', ls='--')
    plt.xlabel(r"Lag time $\tau$")
    plt.ylabel(r"TICA $t_i(\tau)$")
    plt.title(f_str)
    plt.savefig(msm_savedir + "/tica_its_vs_lag.pdf")
    plt.savefig(msm_savedir + "/tica_its_vs_lag.png")

    # cumulative variance
    plt.figure()
    for i in range(len(tica_lags)):
        plt.plot(np.arange(1, len(all_cumvar[i]) + 1), all_cumvar[i], label=str(tica_lags[i]))

    plt.legend(loc=4)
    plt.grid(True, alpha=1, color='k', ls='--')
    #ymin, ymax = plt.ylim()
    plt.ylim(0, 1)
    plt.xlabel("Index")
    plt.ylabel("Kinetic Variance")
    plt.title(f_str)
    plt.savefig(msm_savedir + "/tica_cumvar.pdf")
    plt.savefig(msm_savedir + "/tica_cumvar.png")

    # times vs index
    plt.figure()
    for i in range(len(tica_lags)):
        plt.plot(all_tica_ti[i,:20], 'o', label=str(tica_lags[i]))

    plt.legend()
    plt.grid(True, alpha=1, color='k', ls='--')
    #ymin, ymax = plt.ylim()
    #plt.ylim(ymin, ymax)
    plt.xlabel("Index")
    plt.ylabel(r"TICA $t_i$")
    plt.title(f_str)
    plt.savefig(msm_savedir + "/tica_its.pdf")
    plt.savefig(msm_savedir + "/tica_its.png")

    # determine from TICA cumulative kinetic variance
    #for i in range(len(tica_lags)):
    #    keep_dims = np.argmin((all_cumvar[i] - 0.8)**2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--lagtime", type=int, default=50)
    parser.add_argument("--keep_dims", type=int, default=5)
    parser.add_argument("--use_dihedrals", action="store_true")
    parser.add_argument("--use_distances", action="store_true")
    parser.add_argument("--use_inv_distances", action="store_true")
    parser.add_argument("--use_rg", action="store_true")
    parser.add_argument("--resave_tic", action="store_true")
    parser.add_argument("--noplots", action="store_true")
    args = parser.parse_args()

    name = args.name
    lagtime = args.lagtime
    keep_dims = args.keep_dims
    use_dihedrals = args.use_dihedrals
    use_distances = args.use_distances
    use_inv_distances = args.use_inv_distances
    use_rg = args.use_rg
    resave_tic = args.resave_tic
    noplots = args.noplots

    #python ~/code/implicit_force_field/polymer_scripts/calc_tic.py c25 --use_dihedrals --use_distances --use_rg

    # for tanh contact feature
    r0 = 0.4
    widths = 0.1

    # local density feature DOESN'T WORK YET.

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

    if not os.path.exists(msm_savedir):
        os.mkdir(msm_savedir)

    # save the command that was run to make output
    with open(msm_savedir + "/cmd.txt", "a") as fout:
        fout.write("python " + " ".join(sys.argv) + "\n")

    # get trajectory names
    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
    #trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd")

    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd") 
    traj_idxs = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        traj_idxs.append([idx1, idx2])


    # get indices for dihedral angles and pairwise distances
    feat = coor.featurizer(topfile)
    ply_idxs = feat.topology.select("name PL")

    dih_idxs = np.array([[ply_idxs[i], ply_idxs[i + 1], ply_idxs[i + 2], ply_idxs[i + 3]] for i in range(len(ply_idxs) - 4) ])
    pair_idxs = []
    for i in range(len(ply_idxs) - 1):
        for j in range(i + 4, len(ply_idxs)):
            pair_idxs.append([ply_idxs[i], ply_idxs[j]])
    pair_idxs = np.array(pair_idxs)

    all_pair_idxs = []
    for i in range(len(ply_idxs)):
        this_residue = []
        for j in range(len(ply_idxs)):
            if np.abs(i - j) > 3:
                if i < j:
                    this_residue.append([ply_idxs[i], ply_idxs[j]])
                elif j < i:
                    this_residue.append([ply_idxs[j], ply_idxs[i]])
                else:
                    pass
        all_pair_idxs.append(this_residue)
    all_pair_idxs = np.array(all_pair_idxs)

    # add dihedrals
    if "dih" in feature_set:
        feat.add_dihedrals(dih_idxs, cossin=True)
    if "tanh" in feature_set:
        feat.add_custom_feature(CustomFeature(tanh_contact, len(pair_idxs), fun_args=(pair_idxs, r0, widths)))
    if "dists" in feature_set:
        feat.add_distances(pair_idxs)
    if "invdists" in feature_set:
        feat.add_inverse_distances(pair_idxs)
    if "rho" in feature_set:
        feat.add_custom_feature(CustomFeature(local_density_feature, len(all_pair_idxs), fun_args=(all_pair_idxs, r0, widths)))
    if "rg" in feature_set:
        feat.add_custom_feature(CustomFeature(rg_feature, 1))

    ymin, ymax = 0, 7000

    reader = coor.source(trajnames, features=feat)

    # Estimate Markov state model
    #tica_lag = 20 
    #keep_dims = 23
    #keep_dims = 23

    if not noplots:
        print "Plotting tica timescales vs lagtime..."
        plot_tica_stuff()

    #tica_lag = 50 # lagtime where TICA timescales are converged 
    #keep_dims = 5 # num dims where cumulative variance reaches ~0.8

    tica = coor.tica(lag=lagtime, stride=1)
    coor.pipeline([reader, tica])
    Y = tica.get_output(dimensions=range(keep_dims))
    np.save(msm_savedir + "/tica_ti.npy", tica.timescales)

    print "Saving tica coordinates..."
    #if not os.path.exists(msm_savedir + "/run_1_TIC_1.npy"):
    for i in range(keep_dims):
        for n in range(len(Y)):
            # save TIC with indices of corresponding traj
            idx1, idx2 = traj_idxs[n]
            tic_saveas = msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, i+1)
            if not os.path.exists(tic_saveas) or resave_tic:
                np.save(tic_saveas, Y[n][:,i])

    raise SystemExit



    fig, axes = plt.subplots(5, 1, sharex=True)
    for i in range(5):
        ax = axes[i]
        ax.plot(Y[0][:10000,i])
        ax.set_ylabel("TIC " + str(i + 1))
    fig.savefig(msm_savedir + "/tic_subplot.pdf")

    # plot histogram of tica coordinates
    fig, axes = plt.subplots(4, 4, figsize=(20,20))
    for i in range(4):
        for j in range(i, 4):
            ax = axes[i][j]
            ax.hist2d(Y[0][:,i], Y[0][:,j + 1], bins=50)

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
    fig.savefig(msm_savedir + "/tic_hist_grid.pdf")

    n_clusters = 300
    msm_lags = [1, 10, 20, 50, 100, 200]

    cluster = coor.cluster_kmeans(k=n_clusters)
    coor.pipeline([reader, tica, cluster])
    its = msm.its(cluster.dtrajs, lags=msm_lags)


    plt.figure()
    mplt.plot_implied_timescales(its)
    plt.title(msm_savedir)
    plt.savefig(msm_savedir + "/its_vs_lag_ylog.pdf")

    #plt.figure()
    #plt.plot(np.arange(1,21), M.timescales()[:20], 'o')
    #ymin, ymax = plt.ylim()
    #plt.ylim(0, ymax)
    #plt.savefig("msm_ti.pdf")
