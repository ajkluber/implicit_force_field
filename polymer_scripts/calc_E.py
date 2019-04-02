from __future__ import print_function
import os
import sys
import glob
import argparse
import numpy as np
import scipy.interpolate
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

from scipy.stats import binned_statistic as bin1d

import mdtraj as md

import implicit_force_field.polymer_scripts.util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs coarse-grain model of polymer')
    parser.add_argument("msm_savedir", type=str)
    parser.add_argument("cg_method", type=str)
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
    parser.add_argument("--recalc", action="store_true")
    parser.add_argument('--coeff_file', type=str, default="rdg_fixed_sigma_cstar.npy", help='Specify file with coefficients.')

    args = parser.parse_args()

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
    lin_pot = args.lin_pot
    fix_back = args.fix_back
    fix_exvol = args.fix_exvol
    recalc = args.recalc
    coeff_file = args.coeff_file
    using_U0 = fix_back or fix_exvol

    print(" ".join(sys.argv))

    n_beads = 25
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1/(kb*T)
    #a_coeff = 0.027
     
    if a_coeff is None:
        fixed_a = False
    else:
        fixed_a = True

    savedir = "avgE_vs_orig_tic1"

    # model properties
    using_D2 = False
    n_cross_val_sets = 5

    coeff = np.load(coeff_file)
    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd") 

    if not os.path.exists(savedir + "/avgE_vs_orig_tic1.npy"):
        pass

    Ename = lambda num1, x, num2: "run_{}/{}_{}.npy".format(num1, x, num2)
    cv_name = lambda num1, num2: "run_{}/orig_tic_{}.npy".format(idx1, idx2)

    Ucg, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(msm_savedir,
            n_beads, M, beta, fix_back, fix_exvol, using_cv, using_D2,
            n_cv_basis_funcs, n_cv_test_funcs, n_pair_gauss, bond_cutoff,
            cv_lin_pot=lin_pot, a_coeff=a_coeff, pair_symmetry=pair_symmetry)

    all_cv = []
    all_E0 = []
    all_E1 = []
    for n in range(len(trajnames)):
        print("traj: " + str(n + 1))
        tname = trajnames[n]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        E_exist = [ os.path.exists(Ename(idx1, x, idx2)) for x in ["E0", "E1"] ]
        #psi_exist = os.path.exists()

        # calculate energy and original TIC for trajectory
        if not np.all(E_exist) or recalc:
            cv_traj = []
            E0_traj = []
            E1_traj = []
            for chunk in md.iterload(tname, top=topfile):
                xyz_chunk = np.reshape(chunk.xyz, (-1, 75))
                cv_chunk = Ucg.calculate_cv(xyz_chunk)

                U0_chunk = Ucg.potential_U0(xyz_chunk, cv_chunk, sumterms=False)
                U1_chunk = np.einsum("k,tk->t", coeff, Ucg.potential_U1(xyz_chunk, cv_chunk))

                #breakpoint()
                cv_traj.append(cv_chunk)
                E0_traj.append(U0_chunk)
                E1_traj.extend(U1_chunk)

            cv_traj = np.concatenate(cv_traj, axis=0)
            E0_traj = np.concatenate(E0_traj, axis=1)
            E1_traj = np.array(E1_traj)

            # save
            np.save(cv_name(idx1, idx2), cv_traj)
            np.save(Ename(idx1, "E0", idx2), E0_traj)
            np.save(Ename(idx1, "E1", idx2), E1_traj)
        else:
            cv_traj = np.load(cv_name(idx1, idx2))
            E0_traj = np.load(Ename(idx1, "E0", idx2))
            E1_traj = np.load(Ename(idx1, "E1", idx2))

        all_cv.append(cv_traj[:,0])
        all_E0.append(np.sum(E0_traj, axis=0))
        all_E1.append(E1_traj)

    print("histogramming...")
    # histogram E vs orig_tic
    E = np.concatenate(all_E0) + np.concatenate(all_E1)
    CV = np.concatenate(all_cv)


    avgE, bin_edges, _ = bin1d(CV, E, statistic="mean", bins=40)
    stdE = np.zeros(len(avgE))
    for i in range(len(avgE)):
        frames_in_bin = (CV > bin_edges[i]) & (CV <= bin_edges[i + 1])
        if np.sum(frames_in_bin) > 0:
            stdE[i] = np.sqrt(np.mean((E[frames_in_bin] - avgE[i])**2))

    mid_bin = 0.5*(bin_edges[:-1] + bin_edges[1:])

    if not os.path.exists(savedir):
        os.mkdir(savedir)
    os.chdir(savedir)

    np.save("avgE.npy", avgE)
    np.save("stdE.npy", stdE)
    np.save("mid_bin.npy", mid_bin)

    print("plotting...")
    cwd = os.getcwd()
    plt.figure()
    #plt.title(cwd)
    plt.plot(mid_bin, avgE)
    #plt.fill_between(mid_bin, avgE + stdE, y2=avgE - stdE, alpha=0.2)
    plt.xlabel("Reference TIC $\psi_2$")
    plt.ylabel(r"$\langle E \rangle (\psi_2)$")
    plt.savefig("avgE_vs_orig_tic.pdf")
    plt.savefig("avgE_vs_orig_tic.png")

    n, _ = np.histogram(CV, bins=bin_edges)
    pmf_CV = -(kb*T)*np.log(n.astype(float))
    TS = avgE - pmf_CV

    plt.figure()
    plt.plot(mid_bin, pmf_CV - pmf_CV.min(), 'k', label=r"$F$")
    plt.plot(mid_bin, avgE - avgE.min(), label=r"$\langle E \rangle$")
    plt.plot(mid_bin, TS - avgE.min(), label=r"$TS = \langle E \rangle - F$")
    plt.legend()
    plt.savefig("pmf_E_S_vs_orig_tic.pdf")
    plt.savefig("pmf_E_S_vs_orig_tic.png")


    n, bins = np.histogram(E, bins=40)
    E_mid_bin = 0.5*(bins[1:] + bins[:-1])
    pmf_E = -(kb*T)*np.log(n.astype(float))
    E_TS = E_mid_bin - pmf_E

    plt.figure()
    plt.plot(E_mid_bin, pmf_E - pmf_E.min(), 'k', label=r"$F(E) = -k_BT\ln(P(E))$")
    plt.plot(E_mid_bin, E_mid_bin, '--', label=r"$E$")
    plt.plot(E_mid_bin, E_TS, label=r"$TS = E - F(E)$")
    plt.xlabel(r"$E$")
    plt.legend()
    plt.savefig("pmf_E_S_vs_E.pdf")
    plt.savefig("pmf_E_S_vs_E.png")
