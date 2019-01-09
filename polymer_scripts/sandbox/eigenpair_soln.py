from __future__ import print_function
import os
import sys
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

def plot_train_test_mse(alphas, train_mse, test_mse, 
        xlabel=r"Regularization $\alpha$", ylabel="Mean squared error (MSE)", 
        title="", prefix=""):
    """Plot mean squared error for training and test data"""

    alpha_star = alphas[np.argmin(test_mse[:,0])]

    plt.figure()
    ln1 = plt.plot(alphas, train_mse[:,0], label="Training set")[0]
    ln2 = plt.plot(alphas, test_mse[:,0], label="Test set")[0]
    plt.fill_between(alphas, train_mse[:,0] + train_mse[:,1],
            y2=train_mse[:,0] - train_mse[:,1],
            facecolor=ln1.get_color(), alpha=0.5)

    plt.fill_between(alphas, test_mse[:,0] + test_mse[:,1],
            y2=test_mse[:,0] - test_mse[:,1],
            facecolor=ln2.get_color(), alpha=0.5)

    plt.axvline(alpha_star, color='k', ls='--', label=r"$\alpha^* = {:.2e}$".format(alpha_star))
    plt.semilogx(True)
    plt.semilogy(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(prefix + "train_test_mse.pdf")
    plt.savefig(prefix + "train_test_mse.png")

def plot_Ucg_vs_psi1(coeff, Ucg, cv_r0, prefix):

    U = np.zeros(len(cv_r0))
    for i in range(len(coeff) - 1):
        U += coeff[i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
    U -= U.min()

    plt.figure()
    plt.plot(cv_r0[:,0], U)
    plt.xlabel(r"TIC1 $\psi_1$")
    plt.ylabel(r"$U_{cg}(\psi_1)$")
    plt.savefig("{}U_cv.pdf".format(prefix))
    plt.savefig("{}U_cv.png".format(prefix))

def plot_Ucg_vs_alpha(idxs, idx_star, coeffs, alphas, Ucg, cv_r0, prefix):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for n in range(len(idxs)):

        coeff = coeffs[idxs[n]]

        U = np.zeros(len(cv_r0))
        for i in range(len(coeff) - 1):
            U += coeff[i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
        U -= U.min()

        D = 1./coeff[-1]
        ax1.plot(cv_r0[:,0], U, label=r"$\alpha={:.2e}$".format(alphas[idxs[n]]))
        ax2.plot(cv_r0[:,0], D*np.ones(len(cv_r0[:,0])))

    coeff = coeffs[idx_star]
    U = np.zeros(len(cv_r0))
    for i in range(len(coeff) - 1):
        U += coeff[i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
    U -= U.min()
    D = 1./coeff[-1]
    ax1.plot(cv_r0[:,0], U, color='k', lw=3, label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))
    ax2.plot(cv_r0[:,0], D*np.ones(len(cv_r0[:,0])), color='k', lw=3)
    ax2.semilogy(True)

    ax1.legend()
    ax1.set_xlabel(r"TIC1 $\psi_1$")
    ax1.set_ylabel(r"$U_{cg}(\psi_1)$")
    ax2.set_xlabel(r"TIC1 $\psi_1$")
    ax2.set_ylabel(r"$D$")
    fig.savefig("{}compare_Ucv.pdf".format(prefix))
    fig.savefig("{}compare_Ucv.png".format(prefix))

def split_trajs_into_train_and_test_sets(trajnames, psinames, total_n_frames=None, n_sets=5):
    """Split trajs into sets with roughly same number of frames

    Trajectories are assigned into sets which will be used as training and test
    data sets. For simplicity they are assigned whole, so the number of frames
    in each set will vary.
    
    Parameters
    ----------
    trajnames : list, str
        List of trajectory filenames.

    total_n_frames : int, opt.
        Total number of trajectory frames.

    n_sets : int, default=5
        Desired number of trajectory sets.
        
    """

    if total_n_frames is None:
        traj_n_frames = []
        for n in range(len(trajnames)):
            length = 0
            for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
                length += chunk.n_frames
            traj_n_frames.append(length)
        total_n_frames = sum(traj_n_frames)

    n_frames_in_set = total_n_frames/n_sets

    traj_set = []
    psi_set = []
    traj_set_frames = []
    temp_traj_names = []
    temp_psi_names = []
    temp_frame_count = 0
    for n in range(len(traj_n_frames)):
        if temp_frame_count >= n_frames_in_set:
            # finish a set when it has desired number of frames
            traj_set.append(temp_traj_names)
            psi_set.append(temp_psi_names)
            traj_set_frames.append(temp_frame_count)

            # start over
            temp_traj_names = [trajnames[n]]
            temp_psi_names = [psinames[n]]
            temp_frame_count = traj_n_frames[n]
        else:
            temp_traj_names.append(trajnames[n])
            temp_psi_names.append(psinames[n])
            temp_frame_count += traj_n_frames[n]

        if n == len(traj_n_frames) - 1:
            traj_set.append(temp_traj_names)
            psi_set.append(temp_psi_names)
            traj_set_frames.append(temp_frame_count)

    with open("traj_sets_{}.txt".format(n_sets), "w") as fout:
        for i in range(len(traj_set)):
            info_str = str(traj_set_frames[i])
            info_str += " " + " ".join(traj_set[i]) + "\n"
            fout.write(info_str)

    return traj_set, traj_set_frames, psi_set

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

    n_sets = 5
    trajname_sets, trajname_sets_frames, psiname_sets = split_trajs_into_train_and_test_sets(trajnames, psinames, n_sets=n_sets)

    cg_savedir = cg_savedir + "_crossval_{}".format(n_sets)
    if not os.path.exists(cg_savedir):
        os.mkdir(cg_savedir)

    with open("traj_sets_{}.txt".format(n_sets), "r") as fin:
        with open(cg_savedir + "/traj_sets_{}.txt".format(n_sets), "w") as fout:
            fout.write(fin.read())

    X_chunks = np.all([ os.path.exists("{}/X_{}.npy".format(cg_savedir, i + 1)) for i in range(n_sets) ])
    d_chunks = np.all([ os.path.exists("{}/d_{}.npy".format(cg_savedir, i + 1)) for i in range(n_sets) ])

    if not os.path.exists(cg_savedir + "/X.npy"):
        # cross-validation procedure: calculate coefficients on 
        starttime = time.time()
        print("Iteration:")
        for i in range(len(trajname_sets)):
            Ucg.setup_eigenpair(trajname_sets[i], topfile, psiname_sets[i], ti_file, M=M, cv_names=psiname_sets[i])
            np.save("{}/X_{}.npy".format(cg_savedir, i + 1), Ucg.eigenpair_X)
            np.save("{}/d_{}.npy".format(cg_savedir, i + 1), Ucg.eigenpair_d)
            dt_min = (time.time() - starttime)/60.
            starttime = time.time()
            print("  {}/{} took {} min".format(i+1, len(trajname_sets), dt_min))

        #Ucg.setup_eigenpair(trajnames, topfile, psinames, ti_file, M=M, cv_names=psinames)
        #os.chdir(cg_savedir)
        #np.save("X.npy", Ucg.eigenpair_X)
        #np.save("d.npy", Ucg.eigenpair_d)
    else:
        #os.chdir(cg_savedir)
        #Ucg.eigenpair_X = np.load("X.npy")
        #Ucg.eigenpair_d = np.load("d.npy")
        #X = np.load("X.npy")
        #d = np.load("d.npy")
        X_chunks = [ np.load("{}/X_{}.npy".format(cg_savedir, i + 1)) for i in range(len(trajname_sets)) ]
        d_chunks = [ np.load("{}/d_{}.npy".format(cg_savedir, i + 1)) for i in range(len(trajname_sets)) ]

    total_n_frames = float(sum(trajname_sets_frames)) 
    traj_set_weights = [ trajname_sets_frames[i]/total_n_frames for i in range(len(trajname_sets)) ]

    X_chunks = [ np.load("{}/X_{}.npy".format(cg_savedir, i + 1)) for i in range(len(trajname_sets)) ]
    d_chunks = [ np.load("{}/d_{}.npy".format(cg_savedir, i + 1)) for i in range(len(trajname_sets)) ]

    X = np.sum([ traj_set_weights[j]*X_chunks[j] for j in range(n_sets) ])
    d = np.sum([ traj_set_weights[j]*d_chunks[j] for j in range(n_sets) ])

    # calculate 
    X_sets = []
    d_sets = []
    for i in range(len(trajname_sets)):
        trajname_sets_frames
        w_sum = float(np.sum([ trajname_sets_frames[j] for j in range(n_sets) if j != i ]))
        train_X = np.sum([ (trajname_sets_frames[j]/w_sum)*X_chunks[j] for j in range(n_sets) if j != i ], axis=0)
        train_d = np.sum([ (trajname_sets_frames[j]/w_sum)*d_chunks[j] for j in range(n_sets) if j != i ], axis=0)

        X_sets.append([ train_X, X_chunks[i]])
        d_sets.append([ train_d, d_chunks[i]])

    raise SystemExit

    rdg_alphas = np.logspace(-10, 8, 500)
    rdg_coeffs, rdg_train_mse, rdg_test_mse = iff.util.cross_validated_least_squares(
            rdg_alphas, X, d, np.identity(X.shape[1]), n_splits=50)

    rdg_idx_star = np.argmin(rdg_test_mse[:,0])
    rdg_alpha_star = rdg_alphas[rdg_idx_star]
    rdg_cstar = rdg_coeffs[rdg_idx_star]

    print("Plotting ridge...")
    plot_train_test_mse(rdg_alphas, rdg_train_mse, rdg_test_mse, 
            xlabel=r"Regularization $\alpha$", 
            ylabel="Mean squared error (MSE)", 
            title="Ridge regression", prefix="ridge_")

    d2_alphas = np.logspace(-10, 8, 500)

    D2 = np.zeros((101,101), float)
    D2[:100,:100] = np.load("../Ucg_eigenpair_1D/D2.npy")[:100,:100]

    print("D2 regularization...")
    d2_coeffs, d2_train_mse, d2_test_mse = iff.util.cross_validated_least_squares(
            d2_alphas, X, d, D2, n_splits=50)


    print("Plotting D2...")
    plot_train_test_mse(d2_alphas, d2_train_mse, d2_test_mse, 
            xlabel=r"Regularization $\alpha$", 
            ylabel="Mean squared error (MSE)", 
            title="Second deriv penalty", prefix="D2_")

    d2_idx_star = np.argmin(d2_test_mse[:,0])
    d2_alpha_star = d2_alphas[d2_idx_star]
    d2_cstar = d2_coeffs[d2_idx_star]


    #coeff = np.linalg.lstsq(Ucg.eigenpair_X, Ucg.eigenpair_d, rcond=1e-6)[0]
    #Ucg.eigenpair_coeff = d2_cstar

    #plot_Ucg_vs_psi1(d2_cstar, Ucg, cv_r0, "D2_")
    #plot_Ucg_vs_psi1(rdg_cstar, Ucg, cv_r0, "rdg_")

    rdg_idxs = [5, 50, 200, 300]
    d2_idxs = [50, 100, 300, 480]

    plot_Ucg_vs_alpha(d2_idxs, d2_idx_star, d2_coeffs, d2_alphas, Ucg, cv_r0, "D2_")
    plot_Ucg_vs_alpha(rdg_idxs, rdg_idx_star, rdg_coeffs, rdg_alphas, Ucg, cv_r0, "rdg_")


    raise SystemExit

    alphas = np.logspace(-9, 1, 500)
    alpha_star, coeff, all_coeff, res_norm, reg_norm, cv_score = iff.util.solve_ridge(alphas, X, d)

    idxs = [0, 150, 300, 450, 499]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(len(idxs)):
        ridge = sklin.Ridge(alpha=alphas[idxs[i]], fit_intercept=False)
        ridge.fit(X,d)
        coeff = ridge.coef_

        D = 1./coeff[-1]
        print(str(D))

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

