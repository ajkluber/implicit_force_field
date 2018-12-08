import os
import sys
import glob
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt
import sympy 
x_sym = sympy.symbols("x")

import sklearn.linear_model as sklin

import implicit_force_field as iff

if __name__ == "__main__":
    n_beads = 25
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)

    #msm_savedir = "msm_dih_dists"
    msm_savedir = "msm_dists"

    M = 1   # number of eigenvectors to use


    psi_hist = np.load(msm_savedir + "/psi1_n.npy")
    cv_r0 = np.load(msm_savedir + "/psi1_mid_bin.npy")
    cv_w = np.abs(cv_r0[1] - cv_r0[0])*np.ones(len(cv_r0), float)
    #cv_r0 = np.array([ [cv_r0[i]] for i in range(len(cv_r0)) ])
    #cv_r0 = cv_r0.reshape((len(cv_r0),1))

    print("creating Ucg...")
    # coarse-grain polymer potential with free parameters
    Ucg = iff.basis_library.OneDimensionalModel(1)
    Ucg.add_Gaussian_drift_basis(cv_r0, cv_w)
    Ucg.add_Gaussian_test_functions(cv_r0, cv_w)

    n_b = len(Ucg.b_funcs[1])
    R = n_b                 # number of free model parameters
    P = len(Ucg.f_funcs)    # number of test functions

    ##########################################################
    # EIGENPAIR MATRIX ELEMENTS. VARIABLE DIFFUSION COEFF
    ########################################################## 
    #topfile = glob.glob("run_{}/".format(run_idx) + name + "_min_cent.pdb")[0]
    #trajnames = glob.glob("run_{}/".format(run_idx) + name + "_traj_cent_*.dcd") 
    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd") 
    traj_idxs = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        traj_idxs.append([idx1, idx2])

    kappa = 1./np.load(msm_savedir + "/tica_ti.npy")[0]

    X = np.zeros((P, R+1), float)
    d = np.zeros(P, float)

    print("calculating matrix elements...")
    Ntot = 0
    for n in range(len(traj_idxs)):
        print("traj: " + str(n+1))
        sys.stdout.flush()
        idx1, idx2 = traj_idxs[n]
        Psi = np.load(msm_savedir + "/run_{}_{}_TIC_1.npy".format(idx1, idx2))

        # matrix elements 
        b1 = Ucg.evaluate_parametric_drift(Psi)

        test_f = Ucg.test_functions(Psi)
        grad_f = Ucg.gradient_test_functions(Psi) 
        Lap_f = Ucg.laplacian_test_functions(Psi) 

        temp_X1 = np.einsum("t,tr,tp->pr", Psi, b1, grad_f)
        temp_X2 = (-1/beta)*np.einsum("t,tp->p", Psi, Lap_f)

        temp_d = kappa*np.einsum("t,tp->p", Psi, test_f)

        X[:,:n_b] += temp_X1
        X[:,-1] += temp_X2
        d += temp_d

        Ntot += Psi.shape[0]

    X /= float(Ntot)
    d /= float(Ntot)
    
    #"Ucg_eigenpair"
    cg_savedir = "Ucg_eigenpair_1D_constD"
    if not os.path.exists(cg_savedir):
        os.mkdir(cg_savedir)
    os.chdir(cg_savedir)

    np.save("X.npy", X)
    np.save("d.npy", d)

    with open("X_cond.dat", "w") as fout:
        fout.write(str(np.linalg.cond(X)))

    with open("Ntot.dat", "w") as fout:
        fout.write(str(Ntot))

    lstsq_soln = np.linalg.lstsq(X, d, rcond=1e-8)
    np.save("coeff.npy", lstsq_soln[0])

    pmf = -np.log(psi_hist)
    pmf -= pmf.min()

    # PLAIN LEAST SQUARES
    coeff = lstsq_soln[0]
    b_coeff = coeff[:n_b]
    A = coeff[-1]

    #r = np.linspace(min(cv_r0[2:]), max(cv_r0[:-2]), 200)
    r = np.linspace(min(cv_r0), max(cv_r0), 200)
    #r = np.linspace(-1.1, 1.1, 200)

    drift = np.zeros(len(r), float)
    for i in range(len(b_coeff)):
        drift += b_coeff[i]*Ucg.b_scale_factors[1][i]*Ucg.b_funcs[1][i](r)

    noise = coeff[-1]*np.ones(len(r), float)

    xmin, xmax = min(cv_r0), max(cv_r0)
    ymin, ymax = -2e-2, 2e-2

    # PLOT PMF, DRIFT, AND NOISE
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 15))
    ax1.plot(cv_r0, pmf)
    ax1.set_ylabel(r"Traj PMF $-\log(P(\psi_1))$")
    ax1.set_ylim(0, 4)

    ax2.plot(r, drift)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_ylabel(r"Drift $b(\psi_1)$")

    ax3.plot(r, noise)
    ax3.set_xlim(xmin, xmax)
    ax3.set_xlabel(r"TIC1 $\psi_1$")
    ax3.set_ylabel(r"Noise $a(\psi_1)$")

    fig.savefig("drift_noise_1D_100.pdf")
    fig.savefig("drift_noise_1D_100.png")

    raise SystemExit

    X_msk = np.ma.array(X, mask=(X == 0))

    # PLOT MATRIX ELEMENTS
    fig, ax1 = plt.subplots(1,1, figsize=(6,5))
    cl1 = ax1.pcolormesh(X_msk, linewidth=0, rasterized=True)
    ax1.set_aspect(1)
    ax1.set_xlabel(r"$c_k$")
    ax1.set_ylabel("Test functions")
    fig.colorbar(cl1, ax=ax1)
    fig.savefig("X_matrix_100.pdf")
    fig.savefig("X_matrix_100.png")


    plt.figure()
    plt.plot(d, 'ko')
    plt.xlabel(r"Test function $f_j$")
    plt.ylabel(r"$-\kappa_1\langle \psi_1, f_j\rangle$")
    plt.savefig("d_vector_100.pdf")
    plt.savefig("d_vector_100.png")

    raise SystemExit
    # PLOT PRECONDITIONED MATRIX ELEMENTS
    X_b = pre_X[:,:n_b]
    X_b = np.ma.array(X_b, mask=(X_b == 0))

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))
    cl1 = ax1.pcolormesh(X_b, linewidth=0, rasterized=True)
    ax1.set_aspect(1)
    ax1.set_xlabel(r"Drift $c_k$")
    ax1.set_ylabel("Test functions")
    fig.colorbar(cl1, ax=ax1)

    cl2 = ax2.pcolormesh(X_a, linewidth=0, rasterized=True)
    ax2.set_xlabel(r"Noise $c_k$")
    ax2.set_aspect(1)

    fig.colorbar(cl2, ax=ax2)
    #cl1.get_clim)
    fig.savefig("pre_X_matrix_100.pdf")
    fig.savefig("pre_X_matrix_100.png")


    plt.figure()
    plt.plot(pre_d, 'ko')
    plt.xlabel(r"Test function $f_j$")
    plt.ylabel(r"$-\kappa_1\langle \psi_1, f_j\rangle$")
    plt.savefig("pre_d_vector_100.pdf")
    plt.savefig("pre_d_vector_100.png")


    # PLOT COEFFICIENTS FROM RIDGE ESTIMATOR
    alphas = np.logspace(-16, -2, 400)
    coeff = []
    res_norm = []
    soln_norm = []
    for i in range(len(alphas)):
        ridge = sklin.Ridge(alpha=alphas[i], fit_intercept=False)
        ridge.fit(pre_X,pre_d)
        coeff.append(d2*ridge.coef_)

        res_norm.append(np.linalg.norm(ridge.predict(pre_X) - pre_d))
        soln_norm.append(np.linalg.norm(ridge.coef_))
    
    fig, ax = plt.subplots(1,1)
    ax.plot(alphas, coeff)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1]) 
    ax.set_xlabel(r"Ridge parameter $\alpha$")
    ax.set_ylabel(r"Coefficients $c_k$")
    fig.savefig("ridge_coeff_vs_alpha_100.pdf")
    fig.savefig("ridge_coeff_vs_alpha_100.png")

    plt.figure()
    plt.plot(res_norm, soln_norm)
    plt.semilogx(True)
    plt.semilogy(True)
    plt.xlabel(r"$\log\left(||X\hat{c} - d||_2\right)$")
    plt.ylabel(r"$\log\left(||\hat{c}||_2\right)$")
    plt.savefig("ridge_Lcurve_100.pdf")
    plt.savefig("ridge_Lcurve_100.png")

    plt.figure()
    plt.plot(res_norm, soln_norm)
    #plt.semilogx(True)
    #plt.semilogy(True)
    plt.xlabel(r"$||X\hat{c} - d||_2$")
    plt.ylabel(r"$||\hat{c}||_2$")
    plt.savefig("ridge_Lcurve_100_nolog.pdf")
    plt.savefig("ridge_Lcurve_100_nolog.png")


    # RIDGE HAND-SELECTED ALPHA
    ridge = sklin.Ridge(alpha=1e-5, fit_intercept=False)
    ridge.fit(pre_X,pre_d)
    ck = d2*ridge.coef_

    r = np.linspace(min(cv_r0[2:]), max(cv_r0[:-2]), 200)
    drift, noise, d_noise, dF = calculate_drift_noise(r, Ucg, ck[:n_b], -ck[n_b:])

    xmin, xmax = min(cv_r0), max(cv_r0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))

    ax1.plot(r, drift)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylabel(r"Drift $b(\psi_1)$")

    ax2.plot(r, noise)
    ax2.set_xlim(xmin, xmax)
    ax2.set_xlabel(r"TIC1 $\psi_1$")
    ax2.set_ylabel(r"Noise $a(\psi_1)$")

    fig.suptitle(r"Ridge $\alpha={:.3e}$".format(ridge.alpha))

    fig.savefig("ridge_drift_noise_1D_100.pdf")
    fig.savefig("ridge_drift_noise_1D_100.png")

    # CROSS VALIDATED RIDGE 
    ridge = sklin.RidgeCV(alphas=alphas, cv=5, fit_intercept=False)
    ridge.fit(pre_X,pre_d)
    ck = d2*ridge.coef_

    r = np.linspace(min(cv_r0[2:]), max(cv_r0[:-2]), 200)
    drift, noise, d_noise, dF = calculate_drift_noise(r, Ucg, ck[:n_b], -ck[n_b:])

    xmin, xmax = min(cv_r0), max(cv_r0)

    # PLOT PMF, DRIFT, AND NOISE
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))

    ax1.plot(r, drift)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylabel(r"Drift $b(\psi_1)$")

    ax2.plot(r, noise)
    ax2.set_xlim(xmin, xmax)
    ax2.set_xlabel(r"TIC1 $\psi_1$")
    ax2.set_ylabel(r"Noise $a(\psi_1)$")

    fig.suptitle(r"Ridge CV $\alpha={:.3e}$".format(ridge.alpha_))

    fig.savefig("ridgecv_drift_noise_1D_100.pdf")
    fig.savefig("ridgecv_drift_noise_1D_100.png")
