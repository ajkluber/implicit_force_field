from __future__ import print_function, absolute_import
import os
import glob
import argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

import sklearn.linear_model as sklin

import mdtraj as md

import simulation.openmm as sop

import implicit_force_field as iff
import implicit_force_field.polymer_scripts.util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("msm_savedir", type=str)
    parser.add_argument("--psi_dims", type=int, default=1)
    parser.add_argument("--a_coeff", type=float, default=None)
    parser.add_argument("--n_basis", type=int, default=40)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--fixed_bonds", action="store_true")
    parser.add_argument("--recalc", action="store_true")
    args = parser.parse_args()

    msm_savedir = args.msm_savedir
    M = args.psi_dims
    a_coeff = args.a_coeff
    n_cv_basis_funcs = args.n_basis
    n_cv_test_funcs = args.n_test
    fixed_bonded_terms = args.fixed_bonds
    recalc = args.recalc

    n_beads = 25
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)

    using_D2 = False
    using_cv = True
    using_cv_r0 = False

    n_cross_val_sets = 5

    Ucg, cg_savedir, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(
            msm_savedir, n_beads, M, beta, fixed_bonded_terms, using_cv,
            using_cv_r0, using_D2, n_cv_basis_funcs, n_cv_test_funcs)

    cg_savedir = cg_savedir + "_crossval_{}".format(n_cross_val_sets)

    if not a_coeff is None:
        cg_savedir += "_fixed_a"

    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd") 
    ti_file = msm_savedir + "/tica_ti_ps.npy"
    psinames = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        temp_names = []
        for n in range(M):
            temp_names.append(msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, n+1))
        psinames.append(temp_names)

    all_files_exist = np.all([ os.path.exists(cg_savedir + "/" + xfile) for xfile in 
        ["psi_fj.npy", "psi_gradU0_fj.npy", "psi_gradU1_fj.npy", "psi_Lap_fj.npy", "psi_Gen_fj.npy"] ])

    coeff = np.load(cg_savedir + "/rdg_cstar.npy")

    if not all_files_exist or recalc:
        print("applying generator to f_j...")
        Ucg._generator_scalar_products(coeff, trajnames, topfile, psinames,
                M=1, cv_names=psinames, verbose=True, a_coeff=a_coeff)

        np.save(cg_savedir + "/psi_fj.npy", Ucg._psi_fj)
        np.save(cg_savedir + "/psi_gradU0_fj.npy", Ucg._psi_gU0_fj)
        np.save(cg_savedir + "/psi_gradU1_fj.npy", Ucg._psi_gU1_fj)
        np.save(cg_savedir + "/psi_Lap_fj.npy", Ucg._psi_Lap_fj)
        np.save(cg_savedir + "/psi_Gen_fj.npy", Ucg._psi_Gen_fj)

        psi_fj = Ucg._psi_fj
        psi_gU0_fj = Ucg._psi_gU0_fj
        psi_gU1_fj = Ucg._psi_gU1_fj
        psi_Lap_fj = Ucg._psi_Lap_fj
        psi_Gen_fj = Ucg._psi_Gen_fj
    else:
        psi_fj = np.load(cg_savedir + "/psi_fj.npy")
        psi_gU0_fj = np.load(cg_savedir + "/psi_gradU0_fj.npy")
        psi_gU1_fj = np.load(cg_savedir + "/psi_gradU1_fj.npy")
        psi_Lap_fj = np.load(cg_savedir + "/psi_Lap_fj.npy")
        psi_Gen_fj = np.load(cg_savedir + "/psi_Gen_fj.npy")

#    if not os.path.exists(cg_savedir + "/Lap_fj.npy") or recalc:
#        Ucg._eigenpair_Lap_f(coeff, trajnames, topfile, psinames, M=1, cv_names=psinames, verbose=True)
#
#        np.save(cg_savedir + "/Lap_fj.npy", Ucg.eigenpair_Lap_f)
#        Lap_fj = Ucg.eigenpair_Lap_f
#    else:
#        Lap_fj = np.load(cg_savedir + "/Lap_fj.npy")

    kappa = 1./np.load(ti_file)[:M]

    print("plotting...")
    scale = 10000
    plt.figure()
    plt.plot(cv_r0_test[:,0], scale*psi_Gen_fj[0,:], 'r', label=r"$\langle \psi_1| \mathcal{L} f_j \rangle$")
    plt.plot(cv_r0_test[:,0], -kappa*scale*psi_fj[0,:], 'k', label=r"$-\kappa_1\langle \psi_1| f_j \rangle$")

    plt.xlabel("$f_j$ center along $\psi_1$")
    plt.ylabel("x" + str(scale))
    plt.ylim(-12, 7.5)
    plt.legend(fontsize=14)
    plt.savefig(cg_savedir + "/compare_Gen_fj_psi_fj.pdf")
    plt.savefig(cg_savedir + "/compare_Gen_fj_psi_fj.png")

    plt.figure()
    plt.plot(scale*psi_Gen_fj[0,:], -kappa*scale*psi_fj[0,:], 'ro')

    plt.xlabel(r"$\langle \psi_1| \mathcal{L} f_j \rangle$")
    plt.ylabel(r"$-\kappa_1\langle \psi_1| f_j \rangle$")

    #plt.ylim(-12, 7.5)
    #plt.legend(fontsize=14)
    plt.savefig(cg_savedir + "/scatter_Gen_fj_psi_fj.pdf")
    plt.savefig(cg_savedir + "/scatter_Gen_fj_psi_fj.png")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5,15))
    ax1.plot(cv_r0_test[:,0], psi_gU0_fj[0,:])
    ax2.plot(cv_r0_test[:,0], psi_gU1_fj[0,:])
    ax3.plot(cv_r0_test[:,0], psi_Lap_fj[0,:])

    ax3.set_xlabel("$f_j$ center along $\psi_1$")
    ax1.set_ylabel(r"$\langle \psi_1| -a\nabla U_0 \cdot\nabla f_j \rangle$")
    ax2.set_ylabel(r"$\langle \psi_1| -a\nabla U_1 \cdot\nabla f_j \rangle$")
    ax3.set_ylabel(r"$\langle \psi_1| \Delta f_j \rangle$")

    fig.savefig(cg_savedir + "/compare_psi_gradU_fj.pdf")
    fig.savefig(cg_savedir + "/compare_psi_gradU_fj.png")

    raise SystemExit

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    ax1.plot(cv_r0_test, L_fj)
    ax1.set_xlabel("$f_j$ center along $\psi_1$")
    ax1.set_ylabel(r"$\langle \mathcal{L}(c^*), f_j \rangle$")
    title = r"Testing that $\langle \mathcal{L}(c^*), f_j \rangle = \langle \psi_1, f_j \rangle$"
    ax1.annotate(title, xy=(0,0), xytext=(1.1, 1.2), 
            textcoords="axes fraction", xycoords="axes fraction")

    ax2.plot(cv_r0_test, psi_fj[0,:])
    ax2.set_xlabel("$f_j$ center along $\psi_1$")
    ax2.set_ylabel(r"$\langle \psi_1, f_j \rangle$")
    fig.savefig(cg_savedir + "/compare_L_fj_to_psi_fj.pdf")
    fig.savefig(cg_savedir + "/compare_L_fj_to_psi_fj.png")

    plt.figure()
    plt.plot(cv_r0_test, Lap_fj)
    plt.xlabel("$f_j$ center along $\psi_1$")
    plt.ylabel(r"$\langle \frac{D}{\beta}\Delta f_j \rangle$")
    fig.savefig(cg_savedir + "/compare_Lap_fj.pdf")
    fig.savefig(cg_savedir + "/compare_Lap_fj.png")
