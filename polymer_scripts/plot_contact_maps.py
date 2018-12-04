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

import pyemma.coordinates as coor
import mdtraj as md

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("--keep_dims", type=int, default=5)
    parser.add_argument("--use_dihedrals", action="store_true")
    parser.add_argument("--use_distances", action="store_true")
    parser.add_argument("--use_inv_distances", action="store_true")
    parser.add_argument("--use_rg", action="store_true")
    args = parser.parse_args()

    name = args.name
    keep_dims = args.keep_dims
    use_dihedrals = args.use_dihedrals
    use_distances = args.use_distances
    use_inv_distances = args.use_inv_distances
    use_rg = args.use_rg

    #python ~/code/implicit_force_field/polymer_scripts/plot_contact_maps.py c25 --use_dihedrals --use_distances --keep_dims 5

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

    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd") 
    traj_idxs = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        traj_idxs.append([idx1, idx2])

    print "loading tica..."
    tics = [] 
    for i in range(keep_dims):
        temp_tic = []
        for n in range(len(traj_idxs)):
            idx1, idx2 = traj_idxs[n]

            # save TIC with indices of corresponding traj
            tic_saveas = msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, i+1)
            temp_tic.append(np.load(tic_saveas))
        tics.append(temp_tic)

    all_rg = []
    for n in range(len(traj_idxs)):
        idx1, idx2 = traj_idxs[n]
        all_rg.append(np.load("run_{}/rg_{}.npy".format(idx1, idx2)))


    # bounds of minima
    rg_bounds = [0.38, 0.49]
    tic2_bounds = [[-1.75, -0.5], [0.8, 1.6]]

    # find frames that are in states
    print "finding indices of each state..."
    A_idxs = []
    B_idxs = []
    for n in range(len(traj_idxs)):
        frames_in_A = (rg_bounds[0] <= all_rg[n]) & (all_rg[n] <= rg_bounds[1]) & (tic2_bounds[0][0] <= tics[1][n]) & (tics[1][n] <= tic2_bounds[0][1])
        frames_in_B = (rg_bounds[0] <= all_rg[n]) & (all_rg[n] <= rg_bounds[1]) & (tic2_bounds[1][0] <= tics[1][n]) & (tics[1][n] <= tic2_bounds[1][1])
        if np.sum(frames_in_A) > 0:
            A_idxs.append(np.argwhere(frames_in_A)[:,0])
        else:
            A_idxs.append([])

        if np.sum(frames_in_B) > 0:
            B_idxs.append(np.argwhere(frames_in_B)[:,0])
        else:
            B_idxs.append([])

    # calculate contact map
    feat = coor.featurizer(topfile)
    ply_idxs = feat.topology.select("resname PLY")

    pair_idxs = []
    for i in range(len(ply_idxs) - 1):
        for j in range(i + 4, len(ply_idxs)):
            pair_idxs.append([ply_idxs[i], ply_idxs[j]])
    pair_idxs = np.array(pair_idxs)

    r0 = 0.4
    width = 0.1

    tanh_contact = lambda rij: 0.5*(np.tanh((r0 - rij)/width) + 1)

    print "calculating contact maps..."
    A_qij = np.zeros(len(pair_idxs), float)
    B_qij = np.zeros(len(pair_idxs), float)
    n_tot_A = 0
    n_tot_B = 0
    for n in range(len(traj_idxs)):
        print "   traj:", trajnames[n]
        traj = md.load(trajnames[n], top=topfile)
        if len(A_idxs[n]) > 0:
            traj[A_idxs[n]].save("run_{}/traj_A_{}.dcd".format(traj_idxs[n][0], traj_idxs[n][1]))
            A_qij += tanh_contact(md.compute_distances(traj[A_idxs[n]], pair_idxs)).sum(axis=0)
            n_tot_A += len(A_idxs[n])

        if len(B_idxs[n]) > 0:
            traj[B_idxs[n]].save("run_{}/traj_B_{}.dcd".format(traj_idxs[n][0], traj_idxs[n][1]))
            B_qij = tanh_contact(md.compute_distances(traj[B_idxs[n]], pair_idxs)).sum(axis=0)
            n_tot_B += len(B_idxs[n])
        break
    raise SystemExit

    avg_qij_A = A_qij/float(n_tot_A)
    avg_qij_B = B_qij/float(n_tot_B)

    n_beads = 25
    C_A = np.zeros((n_beads, n_beads), float)
    C_B = np.zeros((n_beads, n_beads), float)
    for i in range(len(pair_idxs)):
        idx1, idx2 = pair_idxs[i]
        C_A[idx2, idx1] = avg_qij_A[i]
        C_B[idx2, idx1] = avg_qij_B[i]

    C_A_msk = np.ma.array(C_A, mask=C_A == 0)
    C_B_msk = np.ma.array(C_B, mask=C_B == 0)

    print "plotting..."
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    #axes[0].pcolormesh(C_A, vmin=0, vmax=1)
    #axes[1].pcolormesh(C_B, vmin=0, vmax=1)
    axes[0].pcolormesh(C_A_msk)
    axes[1].pcolormesh(C_B_msk)
    #axes[0].colorbar()
    #axes[1].colorbar()

    axes[0].set_title("State A")
    axes[1].set_title("State B")
    fig.savefig(msm_savedir + "/cont_map_A_B.pdf")
    fig.savefig(msm_savedir + "/cont_map_A_B.png")

