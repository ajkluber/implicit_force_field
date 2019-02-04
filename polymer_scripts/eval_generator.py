from __future__ import print_function, absolute_import
import os
import glob
import argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import sklearn.linear_model as sklin

import mdtraj as md

import simulation.openmm as sop

import implicit_force_field as iff
import implicit_force_field.polymer_scripts.util as util

if __name__ == "__main__":
    n_beads = 25
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)
    n_pair_gauss = 10
    M = 1   # number of eigenvectors to use
    #fixed_bonded_terms = False
    fixed_bonded_terms = True

    using_D2 = False
    using_cv = True
    using_cv_r0 = False
    #n_cv_basis_funcs = 40
    #n_cv_test_funcs = 100
    #n_cv_basis_funcs = 40
    #n_cv_test_funcs = 40
    n_cv_basis_funcs = 100
    n_cv_test_funcs = 100
    #n_cv_basis_funcs = 100
    #n_cv_test_funcs = 200

    msm_savedir = "msm_dists"
    n_cross_val_sets = 5
    recalc = False
    recalc_Lap_f = False

    Ucg, cg_savedir, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(
            msm_savedir, n_beads, M, beta, fixed_bonded_terms, using_cv,
            using_cv_r0, using_D2, n_cv_basis_funcs, n_cv_test_funcs)

    cg_savedir = cg_savedir + "_crossval_{}".format(n_cross_val_sets)

    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd") 
    ti_file = msm_savedir + "/tica_ti.npy"
    psinames = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        temp_names = []
        for n in range(M):
            temp_names.append(msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, n+1))
        psinames.append(temp_names)

    print("applying generator to f_j...")
    coeff = np.load(cg_savedir + "/rdg_cstar.npy")
    if not os.path.exists(cg_savedir + "/L_fj.npy") or recalc:
        Ucg._eigenpair_generator_terms(coeff, trajnames, topfile, psinames, M=1, cv_names=psinames, verbose=True)

        np.save(cg_savedir + "/psi_fj.npy", Ucg.eigenpair_psi_fj)
        np.save(cg_savedir + "/L_fj.npy", Ucg.eigenpair_L_fj)
        np.save(cg_savedir + "/gradU0_fj.npy", Ucg.eigenpair_gU0_fj)
        np.save(cg_savedir + "/gradU1_fj.npy", Ucg.eigenpair_gU1_fj)
        np.save(cg_savedir + "/L_fj.npy", Ucg.eigenpair_L_fj)

        L_fj = Ucg.eigenpair_L_fj
        psi_fj = Ucg.eigenpair_psi_fj
    else:
        L_fj = np.load(cg_savedir + "/L_fj.npy")
        psi_fj = np.load(cg_savedir + "/psi_fj.npy")

    if not os.path.exists(cg_savedir + "/Lap_fj.npy") or recalc_Lap_f:
        Ucg._eigenpair_Lap_f(coeff, trajnames, topfile, psinames, M=1, cv_names=psinames, verbose=True)

        np.save(cg_savedir + "/Lap_fj.npy", Ucg.eigenpair_Lap_f)
        Lap_fj = Ucg.eigenpair_Lap_f
    else:
        Lap_fj = np.load(cg_savedir + "/Lap_fj.npy")

    kappa = 1./np.load(msm_savedir + "/tica_ti.npy")

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
