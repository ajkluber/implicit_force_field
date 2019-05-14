from __future__ import print_function
import os
import glob
import sys
import time
import argparse
import numpy as np
import scipy.interpolate
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

import mdtraj as md

import implicit_force_field.polymer_scripts.util as util

def plot_vs_time(tics):
    #n_tics = len(tics)
    #fig, axes = plt.subplots(n_tics, 1, sharex=True)
    #for i in range(n_tics):
    #    ax = axes[i]
    #    ax.plot(tics[i][0][:10000])
    #    ax.set_ylabel("TIC " + str(i + 1))
    #fig.savefig("orig_tic_subplot.pdf")
    #fig.savefig("orig_tic_subplot.png")

    n_tic1 = len(tics[0])
    max_length = np.max([ x.shape[0] for x in tics[0] ])
    fig, axes = plt.subplots(n_tic1, 1, figsize=(15, n_tic1*4), sharex=True)
    for i in range(n_tic1):
        ax = axes[i]
        ax.plot(tics[0][i])
        ax.set_ylabel("run " + str(i + 1))
        ax.set_ylim(-2, 2)
        ax.set_xlim(0, max_length)

    ax.set_xlabel("TIC 1")
    fig.savefig("orig_tic1_subplot.pdf")
    fig.savefig("orig_tic1_subplot.png")


def plot_tic_vs_tic(tics, msm_savedir):

    n_cols = np.min([ len(tics), 3])

    # plot histogram of tica coordinates
    fig, axes = plt.subplots(n_cols, n_cols, figsize=(5*n_cols, 5*n_cols))
    for i in range(n_cols):
        for j in range(i, n_cols):
            ax = axes[i][j]
            x = np.concatenate(tics[i])
            y = np.concatenate(tics[j + 1])

            H, xedges, yedges = np.histogram2d(x, y, bins=100)
            X, Y = np.meshgrid(xedges, yedges)
            Hmsk = np.ma.array(H, mask=H ==0)
            #ax.hist2d(np.concatenate(tics[i]), , bins=100)

            pcol = ax.pcolormesh(X, Y, Hmsk, linewidth=0, rasterized=True)
            pcol.set_edgecolor("face")

            if i == (n_cols - 1):
                ax.set_xlabel("TIC " + str(i + 2), fontsize=20)
            #if j == 0:
            #    ax.set_ylabel("TIC " + str(j + 1), fontsize=20)
            #    #ax.set_title("TIC " + str(i + 2), fontsize=20)

        axes[i][0].set_ylabel("TIC " + str(i + 1), fontsize=20)

        if i == (n_cols - 1):
            for j in range(n_cols):
                axes[i][j].set_xlabel("TIC " + str(j + 2), fontsize=20)

    axes[0][0].annotate("TICA  " + msm_savedir.split("msm_")[1], fontsize=24, xy=(0,0),
            xytext=(1.8, 1.1), xycoords="axes fraction", textcoords="axes fraction")
    fig.savefig("orig_tic_hist_grid.pdf")
    fig.savefig("orig_tic_hist_grid.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs coarse-grain model of polymer')
    parser.add_argument('msm_savedir', type=str, help='MSM save directory.')
    parser.add_argument('cg_method', type=str, help='Coarse-graining method')
    parser.add_argument("--psi_dims", type=int, default=1)
    parser.add_argument("--a_coeff", type=float, default=None, help='Diffusion coefficient used in eigenpair.')
    parser.add_argument("--using_cv", action="store_true")
    parser.add_argument("--n_basis", type=int, default=-1)
    parser.add_argument("--n_test", type=int, default=-1)
    parser.add_argument("--n_pair_gauss", type=int, default=-1)
    parser.add_argument("--pair_symmetry", type=str, default=None)
    parser.add_argument("--bond_cutoff", type=int, default=4)
    parser.add_argument("--lin_pot", action="store_true")
    parser.add_argument("--fix_back", action="store_true")
    parser.add_argument("--fix_exvol", action="store_true")
    parser.add_argument('--coeff_file', type=str, default="rdg_fixed_sigma_cstar.npy", help='Specify file with coefficients.')

    parser.add_argument("--recalc", action="store_true")
    parser.add_argument("--replot", action="store_true")
    parser.add_argument('--T', type=float, default=300, help='Temperature.')
    args = parser.parse_args()

    #python ~/code/implicit_force_field/polymer_scripts/plot_cg_tics.py msm_dists eigenpair --psi_dims 1 --using_cv --lin_pot --n_basis 40 --n_test 100 --a_coeff 0.000135 --fix_back --bond_cutoff 4
    
    n_beads = 25
    name = "c" + str(n_beads)
    #name = args.name
    #n_beads = args.n_beads

    msm_savedir = args.msm_savedir
    cg_method = args.cg_method
    M = args.psi_dims
    a_coeff = args.a_coeff
    using_cv = args.using_cv
    n_cv_basis_funcs = args.n_basis
    n_cv_test_funcs = args.n_test
    n_pair_gauss = args.n_pair_gauss
    pair_symmetry = args.pair_symmetry
    bond_cutoff = args.bond_cutoff
    fix_back = args.fix_back
    fix_exvol = args.fix_exvol
    lin_pot = args.lin_pot

    recalc = args.recalc
    replot = args.replot
    T = args.T
    beta = 1/(0.0083145*T)

    using_U0 = fix_back or fix_exvol

    print(" ".join(sys.argv))

    using_D2 = False
    n_cross_val_sets = 5

    if (n_cv_basis_funcs != -1) and (n_cv_test_funcs != -1):
        print("Since n_test ({}) and n_basis ({}) are specified -> using_cv=True".format(n_cv_test_funcs, n_cv_basis_funcs))
        using_cv = True
    else:
        if using_cv:
            raise ValueError("Please specify n_test and n_basis")

    cg_savedir = util.Ucg_dirname(cg_method, M, using_U0, fix_back, fix_exvol,
            bond_cutoff, using_cv, n_cv_basis_funcs=n_cv_basis_funcs,
            n_cv_test_funcs=n_cv_test_funcs, a_coeff=a_coeff,
            n_pair_gauss=n_pair_gauss, cv_lin_pot=lin_pot,
            pair_symmetry=pair_symmetry)

    print(cg_savedir)
    n_tics = 4

    #Ucg, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(
    #        msm_savedir, n_beads, M, beta, fix_back, fix_exvol, using_cv,
    #        using_D2, n_cv_basis_funcs, n_cv_test_funcs, n_pair_gauss,
    #        bond_cutoff, a_coeff=a_coeff, n_cvs=n_tics)

    Ucg = util.create_Ucg_collective_variable(msm_savedir, n_beads, n_tics,
            beta, using_cv, using_D2, bond_cutoff, a_coeff=a_coeff)

    cwd = os.getcwd()
    Hdir = cwd + "/" + cg_savedir

    rundir_str = lambda idx: Hdir + "/run_{}".format(idx)

    os.chdir(cg_savedir)

    # in the coarse-grain simulation directory.
    cg_trajnames = glob.glob("run_*/{}_traj_*.dcd".format(name))
    cg_topname = glob.glob("run_*/{}_noslv_min.pdb".format(name))[0]

    ticname = lambda idx1, idx2, idx3: "run_{}/traj_{}_orig_TIC_{}.npy".format(idx1, idx2, idx3)

    #recalc = True
    tics = [ [] for n in range(n_tics) ]
    for i in range(len(cg_trajnames)):
        tname = cg_trajnames[i] 
        run_idx = os.path.dirname(tname).split("_")[1]
        traj_idx = (os.path.basename(tname).split("_")[-1]).split(".dcd")[0]

        psi_files_exist = [ os.path.exists(ticname(run_idx, traj_idx, n + 1)) for n in range(n_tics) ]
        #breakpoint()
        if np.all(psi_files_exist) and not recalc:
            print("loading tics...")
            # load
            for n in range(n_tics):
                tics[n].append(np.load(ticname(run_idx, traj_idx, n + 1)))
        else:
            print("calculating tics...")
            last_change = np.abs(os.path.getmtime(tname) - time.time())/60.

            # only use if trajectory hasn't been changed in 5 minutes,
            # otherwise it might still be being generated.
            if last_change > 5:
                tic_temp = []
                for chunk in md.iterload(cg_trajnames[i], top=cg_topname):
                    xyz_chunk = np.reshape(chunk.xyz, (-1, Ucg.n_dof))
                    tic_chunk = Ucg.calculate_cv(xyz_chunk)
                    tic_temp.append(tic_chunk)
                tic_traj = np.concatenate(tic_temp)

                for n in range(n_tics):
                    # save each TIC
                    np.save(ticname(run_idx, traj_idx, n + 1), tic_traj[:,n])
                    tics[n].append(tic_traj[:,n])

    tic1 = np.concatenate(tics[0])

    print(str(tic1.shape[0]))

    n, bins = np.histogram(tic1, bins=40)
    mid_bin = 0.5*(bins[1:] + bins[:-1])
    n = n.astype(float)
    n /= np.sum(n)

    pmf = -np.log(n.astype(float))
    pmf -= pmf.min()

    plt.figure()
    plt.plot(mid_bin, n, 'k-')
    plt.xlabel("TIC 1 $\psi_1$")
    plt.ylabel("Probability")
    plt.savefig("orig_psi1_hist.pdf")
    plt.savefig("orig_psi1_hist.png")

    plt.figure()
    plt.plot(mid_bin, pmf, 'k-')
    plt.xlabel("TIC 1 $\psi_1$")
    plt.ylabel("$-\ln P(\psi_1)$   (k$_B$T)")
    plt.savefig("orig_psi1_pmf.pdf")
    plt.savefig("orig_psi1_pmf.png")

    if not os.path.exists("orig_tic_hist_grid.pdf") or replot:
        plot_tic_vs_tic(tics, msm_savedir)

    if not os.path.exists("orig_tic1_subplot.pdf") or replot:
        plot_vs_time(tics)
