import os
import sys
from __future__ import print_function
import glob
import time
import argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import sklearn.linear_model as sklin

import simtk.unit as unit
import simtk.openmm.app as app
import mdtraj as md

import simulation.openmm as sop
import implicit_force_field as iff

if __name__ == "__main__":
    n_beads = 25
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)
    n_pair_gauss = 10
    M = 1   # number of eigenvectors to use
    fixed_bonded_terms = False
    using_cv = True
    using_D2 = False

    #msm_savedir = "msm_dih_dists"
    msm_savedir = "msm_dists"

    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    r0, kb, theta0, ka = bonded_params
    app.element.polymer = app.element.Element(200, "Polymer", "Pl", mass_ply)

    sigma_ply_nm = sigma_ply/unit.nanometer
    #r0_wca_nm = sigma_ply_nm*(2**(1./6))
    eps_ply_kj = eps_ply/unit.kilojoule_per_mole
    kb_kj = kb/(unit.kilojoule_per_mole/(unit.nanometer**2))
    ka_kj = (ka/(unit.kilojoule_per_mole/(unit.radian**2)))
    theta0_rad = theta0/unit.radian
    r0_nm = r0/unit.nanometer

    gauss_r0_nm = np.linspace(0.3, 1, n_pair_gauss)
    gauss_sigma = gauss_r0_nm[1] - gauss_r0_nm[0]
    gauss_w_nm = gauss_sigma*np.ones(len(gauss_r0_nm))

    print("creating Ucg...")
    # coarse-grain polymer potential with free parameters
    Ucg = iff.basis_library.PolymerModel(n_beads, beta, using_cv=using_cv, using_D2=using_D2)
    cg_savedir = "Ucg_eigenpair"

    if fixed_bonded_terms:
        cg_savedir += "_fixed_bonds_angles"
        Ucg.harmonic_bond_potentials(r0_nm, scale_factor=kb_kj, fixed=True)
        Ucg.harmonic_angle_potentials(theta0_rad, scale_factor=ka_kj, fixed=True)
        #Ucg.LJ6_potentials(sigma_ply_nm, scale_factor=eps_ply_kj)
        Ucg.inverse_r12_potentials(sigma_ply_nm, scale_factor=0.5, fixed=True)

    if using_cv:
        # centers of test functions in collective variable (CV) space
        ply_idxs = np.arange(n_beads)
        pair_idxs = []
        for i in range(len(ply_idxs) - 1):
            for j in range(i + 4, len(ply_idxs)):
                pair_idxs.append([ply_idxs[i], ply_idxs[j]])
        pair_idxs = np.array(pair_idxs)

        cv_r0 = np.load(msm_savedir + "/psi1_mid_bin.npy")
        cv_w = np.abs(cv_r0[1] - cv_r0[0])*np.ones(len(cv_r0), float)
        cv_r0 = cv_r0.reshape((len(cv_r0),1))

        cv_coeff = np.load(msm_savedir + "/tica_eigenvects.npy")[:,:M]
        cv_mean = np.load(msm_savedir + "/tica_mean.npy")

        Ucg.linear_collective_variables(["dist"], pair_idxs, cv_coeff, cv_mean)
        Ucg.gaussian_cv_test_funcs(cv_r0, cv_w)
        Ucg.gaussian_cv_potentials(cv_r0, cv_w)

        cg_savedir += "_CV_{}_{}".format(M, len(cv_r0))
    else:
        cg_savedir += "_gauss_pairs_{}".format(n_pair_gauss)
        Ucg.gaussian_pair_potentials(gauss_r0_nm, gauss_w_nm, scale_factor=10)
        Ucg.gaussian_bond_test_funcs([r0_nm], [0.3])
        Ucg.vonMises_angle_test_funcs([theta0_rad], [4])
        Ucg.gaussian_pair_test_funcs(gauss_r0_nm, gauss_w_nm)

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

    cg_savedir = "TESTING_" + cg_savedir
    if not os.path.exists(cg_savedir):
        os.mkdir(cg_savedir)

    if not os.path.exists(cg_savedir + "/X.npy"):
        Ucg.setup_eigenpair(trajnames, topfile, psinames, ti_file, M=M, cv_names=psinames)
        os.chdir(cg_savedir)
        np.save("X.npy", Ucg.eigenpair_X)
        np.save("d.npy", Ucg.eigenpair_d)
    else:
        os.chdir(cg_savedir)
        #Ucg.eigenpair_X = np.load("X.npy")
        #Ucg.eigenpair_d = np.load("d.npy")
        X = np.load("X.npy")
        d = np.load("d.npy")

    raise SystemExit

    alphas = np.logspace(-10, -1, 400)
    D2 = np.zeros((101,101), float)
    D2[:100,:100] = np.load("../Ucg_eigenpair_1D/D2.npy")[:100,:100]


    all_coeff, res_norm, deriv_norm, cv_score = iff.util.solve_D2_regularized(
            alphas, X, d, D2, n_b=100, weight_a=None, variable_noise=False)

    raise SystemExit

    coeff = np.linalg.lstsq(Ucg.eigenpair_X, Ucg.eigenpair_d, rcond=1e-6)[0]

    D = 1./coeff[-1]

    U = np.zeros(len(cv_r0))
    for i in range(len(coeff) - 1):
        U += coeff[i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
    U -= U.min()

    plt.figure()
    plt.plot(cv_r0[:,0], U)
    plt.xlabel("TIC1")
    plt.ylabel(r"$U_{cg}$")
    plt.savefig("U_cv.pdf")
    plt.savefig("U_cv.png")

    alphas = np.logspace(-9, 1, 500)
    alpha_star, coeff, all_coeff, res_norm, reg_norm, cv_score = iff.util.solve_ridge(alphas, X, d)

    idxs = [0, 150, 300, 450, 499]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(len(idxs)):
        ridge = sklin.Ridge(alpha=alphas[idxs[i]], fit_intercept=False)
        ridge.fit(X,d)
        coeff = ridge.coef_

        D = 1./coeff[-1]
        print D

        U = np.zeros(len(cv_r0))
        for n in range(len(coeff) - 1):
            U += coeff[n]*Ucg.cv_U_funcs[n](cv_r0[:,0])
        U -= U.min()

        axes[0].plot(cv_r0[:,0], U, label=r"$\alpha={:.2e}$".format(alphas[idxs[i]]))
        #axes[0].plot(cv_r0[:,0], U, label=r"$\log_{10}(\alpha)={:.2f}$".format(np.log10(alphas[idxs[i]])))
        axes[1].plot(cv_r0[:,0], D*np.ones(len(cv_r0[:,0])))
    axes[0].set_xlabel("TIC1")
    axes[0].set_ylabel(r"$U(\psi_1)$")

    axes[1].set_xlabel("TIC1")
    axes[1].set_ylabel(r"$D$")

    axes[1].semilogy(True)
    axes[0].legend(fancybox=False, frameon=True, edgecolor="k", framealpha=1)
    fig.savefig("ridge_compare_solns.pdf")
    fig.savefig("ridge_compare_solns.png")

    # plot just one solution
    plt.figure() 
    ridge = sklin.Ridge(alpha=alphas[-1], fit_intercept=False)
    ridge.fit(X,d)
    coeff = ridge.coef_

    U = np.zeros(len(cv_r0))
    for n in range(len(coeff) - 1):
        U += coeff[n]*Ucg.cv_U_funcs[n](cv_r0[:,0])
    U -= U.min()

    plt.plot(cv_r0[:,0], U, label=r"$\alpha={:.2e}$".format(alphas[idxs[i]]))
    plt.xlabel("TIC1")
    plt.ylabel(r"$U(\psi_1)$")
    plt.legend(fancybox=False, frameon=True, edgecolor="k", framealpha=1)
    plt.savefig("ridge_example_soln.pdf")
    plt.savefig("ridge_example_soln.png")

    raise SystemExit






    raise SystemExit

    traj = md.load(traj_name, top=topname)

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_validate

    import sklearn.model_selection

    cv_mean = []
    cv_std = []
    coeffs = []
    alphas = np.logspace(-10, 3, num=100)
    for i in range(len(alphas)):
        print(i)
        rdg = Ridge(alpha=alphas[i], fit_intercept=False)
        rdg = Ridge(alpha=1, fit_intercept=False)
        #X_train, X_test, y_train, y_test = train_test_split(G, frav, test_size=0.5, random_state=0)
        #model = rdg.fit(X_train, y_train)
        #cv_score.append(model.score(X_test, y_test))
        #coeffs.append(model.coef_)

        #model = rdg.fit(G, frav)
        #coeffs.append(model.coef_)

        #scores = cross_val_score(rdg, G, frav, cv=5)
        cv = cross_validate(rdg, G, frav, cv=5, return_estimator=True)

        cv_mean.append(scores.mean())
        cv_std.append(scores.std())
        

    train_sc, test_sc = sklearn.model_selection.validation_curve(rdg, G, frav, "alpha", np.logspace(-10, 3), cv=5)

    plt.figure()
    plt.errorbar(alphas, cv_mean, yerr=cv_std)

    plt.savefig("cv_score_vs_alpha.pdf")
    plt.savefig("cv_score_vs_alpha.png")

    plt.figure()
    plt.plot(alphas, cv_score)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"CV score")
    plt.savefig("cv_score_vs_alpha.pdf")
    plt.savefig("cv_score_vs_alpha.png")

