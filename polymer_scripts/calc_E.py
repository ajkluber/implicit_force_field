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

def junk():
    cv_traj = []
    for chunk in md.iterload(tname, top=topfile):
        xyz_chunk = np.reshape(chunk.xyz, (-1, 75))
        cv_chunk = Ucg.calculate_cv(xyz_chunk)

        cv_traj.append(cv_chunk)
    cv_traj = np.concatenate(cv_traj, axis=0)


def hist_psi1():
    ticnames = glob.glob("../msm_dists/run_*_TIC_1.npy")

    temp_n = np.zeros(len(bin_edges) - 1)
    for i in range(len(ticnames)):
        temp_tic = np.load(ticnames[i])
        n, bins = np.histogram(temp_tic, bins=bin_edges)
        temp_n += n

    temp_n /= float(np.sum(temp_n))

    plt.figure()
    plt.plot(mid_bin, temp_n, 'k', lw=2)
    plt.xlabel(r"$\psi_2$")
    plt.ylabel(r"$P(\psi_2)$")
    plt.savefig("../msm_dists/temp_psi1_dist.pdf")

    temp_pmf = -np.log(temp_n)
    temp_pmf -= temp_pmf.min()

    plt.figure()
    plt.plot(mid_bin, temp_pmf, 'k', lw=2)
    plt.xlabel(r"$\psi_2$")
    plt.ylabel(r"$-\lnP(\psi_2)$")
    plt.savefig("../msm_dists/temp_psi1_pmf.pdf")

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

    #python ~/code/implicit_force_field/polymer_scripts/calc_E.py ../msm_dists force-matching --psi_dims 1 --a_coeff 0.027 --using_cv --n_basis 40 --n_test 100 --bond_cutoff 4 --fix_back --lin_pot
    #python ~/code/implicit_force_field/polymer_scripts/calc_E.py ../msm_dists eigenpair --psi_dims 1 --a_coeff 0.000135 --using_cv --n_basis 40 --n_test 100 --bond_cutoff 4 --fix_back --lin_pot

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
    deltaT = 0.2

    # model properties
    using_D2 = False
    n_cross_val_sets = 5
    use_cent = False

    coeff = np.load(coeff_file)

    if use_cent:
        trajnames = glob.glob("run_*/" + name + "_traj_cent_[1-9]*.dcd") 
        topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
    else:
        trajnames = glob.glob("run_*/" + name + "_traj_[1-9]*.dcd") 
        topfile = glob.glob("run_*/" + name + "_noslv_min.pdb")[0]

    run_names = np.unique([ os.path.dirname(x) for x in glob.glob("run_*/" + name + "_traj*dcd") ])
    run_idxs = np.sort([ int(y.split("_")[1]) for y in run_names ])

    traj_idxs = []
    for i in range(len(run_idxs)):
        os.chdir("run_" + str(run_idxs[i]))
        if use_cent:
            r_trajnames = glob.glob(name + "_traj_cent_[1-9]*.dcd") 
        else:
            r_trajnames = glob.glob(name + "_traj_[1-9]*.dcd") 

        r_traj_idxs = np.sort([ int((x.split("_")[-1])[:-4]) for x in r_trajnames ])

        if not np.allclose(r_traj_idxs[1:] - r_traj_idxs[:-1], np.ones(len(r_traj_idxs) - 1)):
            print("trajs for run_{} are non-consequetive".format(run_idxs[i]))

        traj_idxs.append(r_traj_idxs)
            
        os.chdir("..")

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
    for i in range(len(run_idxs)):
        idx1 = run_idxs[i]

        cv_run = []
        E0_run = []
        E1_run = []
        for j in range(len(traj_idxs[i])):
            idx2 = traj_idxs[i][j]
            if use_cent:
                tname = "run_{}/{}_traj_cent_{}.dcd".format(idx1, name, idx2)
            else:
                tname = "run_{}/{}_traj_{}.dcd".format(idx1, name, idx2)

            cv_exist = os.path.exists(cv_name(idx1, idx2))
            E_exist = np.all([ os.path.exists(Ename(idx1, x, idx2)) for x in ["E0", "E1"] ])
            files_exist = E_exist and cv_exist

            if not files_exist or recalc:
                print("calculating run_{} traj_{}".format(idx1, idx2), end="\r")
                cv_traj = []
                E0_traj = []
                E1_traj = []
                for chunk in md.iterload(tname, top=topfile):
                    xyz_chunk = np.reshape(chunk.xyz, (-1, 75))
                    cv_chunk = Ucg.calculate_cv(xyz_chunk)

                    U0_chunk = Ucg.potential_U0(xyz_chunk, cv_chunk, sumterms=False)
                    U1_chunk = np.einsum("k,tk->t", coeff, Ucg.potential_U1(xyz_chunk, cv_chunk))

                    cv_traj.append(cv_chunk)
                    E0_traj.append(U0_chunk)
                    E1_traj.extend(U1_chunk)

                cv_traj = np.concatenate(cv_traj, axis=0)
                E0_traj = np.concatenate(E0_traj, axis=1)
                E1_traj = np.array(E1_traj)

                np.save(cv_name(idx1, idx2), cv_traj)
                np.save(Ename(idx1, "E0", idx2), E0_traj)
                np.save(Ename(idx1, "E1", idx2), E1_traj)
            else:
                print("loading run_{} traj_{}".format(idx1, idx2), end="\r")
                cv_traj = np.load(cv_name(idx1, idx2))
                E0_traj = np.load(Ename(idx1, "E0", idx2))
                E1_traj = np.load(Ename(idx1, "E1", idx2))

            cv_run.append(cv_traj[:,0])
            E0_run.append(np.sum(E0_traj, axis=0))
            E1_run.append(E1_traj)

        all_cv.append(np.concatenate(cv_run))
        all_E0.append(np.concatenate(E0_run))
        all_E1.append(np.concatenate(E1_run))

    print(" ")

    fig, axes = plt.subplots(len(run_idxs), 1, figsize=(10, len(run_idxs)*4), sharex=True, sharey=True)
    maxx = 0
    for i in range(len(run_idxs)):
        run_cv = all_cv[i]

        if run_cv.shape[0] > maxx:
            maxx = run_cv.shape[0]
        ax = axes[i]
        x = deltaT*np.arange(0, len(run_cv))
        ax.annotate("Run " + str(run_idxs[i]), xy=(0,0), xytext=(0.02, 0.85),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=18, bbox={"alpha":1, "edgecolor":"k", "facecolor":"w"})
        ax.plot(x, run_cv, lw=0.5)
        ax.set_xlim(0, deltaT*maxx)
        ax.set_ylim(-2, 2)
        ax.set_ylabel("Ref $\psi_2$")

    ax.set_xlabel("Time (ps)")
    fig.savefig("orig_tic1_vs_time.pdf")
    #fig.savefig("orig_tic1_vs_time.png")

    print("histogramming...")
    # histogram E vs orig_tic
    E = np.concatenate(all_E0) + np.concatenate(all_E1)
    CV = np.concatenate(all_cv)

    n, bin_edges = np.histogram(CV, bins=40)
    mid_bin = 0.5*(bin_edges[:-1] + bin_edges[1:])

    plt.figure()
    plt.plot(mid_bin, n, 'k-')
    plt.xlabel(r"Reference $\psi_2$")
    plt.ylabel("Probability")
    plt.savefig("orig_psi1_dist.pdf")
    plt.savefig("orig_psi1_dist.png")

    pmf_CV = -np.log(n.astype(float))
    pmf_CV -= pmf_CV.min()

    plt.figure()
    plt.plot(mid_bin, pmf_CV, 'k-')
    plt.xlabel("TIC 1 $\psi_1$")
    plt.ylabel("$-\ln P(\psi_1)$  ($k_BT$)")
    plt.savefig("orig_psi1_pmf.pdf")
    plt.savefig("orig_psi1_pmf.png")

    avgE, _, _ = bin1d(CV, E, statistic="mean", bins=bin_edges)
    stdE = np.zeros(len(avgE))
    for i in range(len(avgE)):
        frames_in_bin = (CV > bin_edges[i]) & (CV <= bin_edges[i + 1])
        if np.sum(frames_in_bin) > 0:
            stdE[i] = np.sqrt(np.mean((E[frames_in_bin] - avgE[i])**2))

    TS = avgE - kb*T*pmf_CV

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
