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

def calculate_drift_noise(r, Ucg, b_coeff, a_coeff):

    drift = np.zeros(len(r), float)
    for i in range(len(b_coeff)):
        drift += b_coeff[i]*Ucg.b_scale_factors[1][i]*Ucg.b_funcs[1][i](r)

    noise = np.zeros(len(r), float)
    for i in range(len(a_coeff)):
        noise += a_coeff[i]*Ucg.a_scale_factors[1][i]*Ucg.a_funcs[1][i](r)

    d_noise = np.zeros(len(r), float)
    for i in range(len(a_coeff)):
        da_temp = sympy.lambdify(x_sym, Ucg.a_sym[1][i].diff(x_sym), modules="numpy")(r)
        d_noise += a_coeff[i]*Ucg.a_scale_factors[1][i]*da_temp

    dF = (1/noise)*((1/beta)*d_noise - drift)

    return drift, noise, d_noise, dF

def plot_regularization_soln(alphas, coeff, res_norm, reg_norm, ylabel, title, prefix, suffix):
    """Plot coefficients from regularization results"""

    fig, ax = plt.subplots(1,1)
    ax.plot(alphas, coeff)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1]) 
    ax.set_xlabel(r"Regularization $\alpha$")
    ax.set_ylabel(r"Coefficients $c_k$")
    fig.savefig("{}coeff_vs_alpha{}.pdf".format(prefix, suffix))
    fig.savefig("{}coeff_vs_alpha{}.png".format(prefix, suffix))

    plt.figure()
    plt.plot(res_norm, reg_norm)
    plt.xlabel(r"$||X\hat{c} - d||_2$")
    plt.ylabel(r"${}$".format(ylabel))
    plt.savefig("{}Lcurve{}_nolog.pdf".format(prefix, suffix))
    plt.savefig("{}Lcurve{}_nolog.png".format(prefix, suffix))

    plt.figure()
    plt.plot(res_norm, reg_norm)
    plt.semilogx(True)
    plt.semilogy(True)
    plt.xlabel(r"$\log\left(||X\hat{c} - d||_2\right)$")
    plt.ylabel(r"$\log\left({}\right)$".format(ylabel))
    plt.savefig("{}Lcurve{}.pdf".format(prefix, suffix))
    plt.savefig("{}Lcurve{}.png".format(prefix, suffix))

def plot_select_drift_and_noise(alphas, A, b, Ucg, r, emp_r, emp_dF, xlabel,
        title, prefix, suffix, method="ridge", right_precond=None, alim=None, blim=None,
        Flim=None):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))

    n_b = len(Ucg.b_funcs[1])
    n_a = len(Ucg.a_funcs[1])

    ax4.plot(emp_r, -emp_dF, 'k', label="Emp")

    if method == "ridge":
        alpha_star, coeff, all_soln, res_norm, reg_norm = iff.util.solve_ridge(alphas, A, b, right_precond=right_precond)
    elif method == "1st_deriv":
        all_soln, res_norm, reg_norm = iff.util.solve_deriv_regularized(alphas, X, d, Ucg, r, order=1, variable_noise=True)
    elif method == "2nd_deriv":
        all_soln, res_norm, reg_norm = iff.util.solve_deriv_regularized(alphas, X, d, Ucg, r, order=2, variable_noise=True)

    for i in range(len(alphas)):
        # evaluate drift
        b_coeff = all_soln[i][:n_b]
        a_coeff = all_soln[i][n_b:]
        drift, noise, d_noise, dF = calculate_drift_noise(r, Ucg, b_coeff, a_coeff)

        ax1.plot(r, drift, label=r"$\alpha = {:.2e}$".format(alphas[i]))
        ax2.plot(r, noise)
        ax4.plot(r, -dF)

        # compare the mean force from 
        drift, noise, d_noise, dF = calculate_drift_noise(emp_r, Ucg, b_coeff, a_coeff)
        emp_drift = -noise*emp_dF + d_noise/beta
          
        ax3.plot(emp_r, emp_drift, label=r"$\alpha = {:.2e}$".format(alphas[i]))

    ax1.legend(loc=2, fancybox=False, edgecolor="k", facecolor="w", framealpha=1)
    ax4.legend(loc=2, fancybox=False, edgecolor="k", facecolor="w", framealpha=1)

    #ax1.set_xlabel(xlabel)
    #ax2.set_xlabel(xlabel)
    ax3.set_xlabel(xlabel)
    ax4.set_xlabel(xlabel)

    ax1.set_ylabel("Drift")
    ax2.set_ylabel("Noise")
    ax3.set_ylabel("Emp. Drift")
    ax4.set_ylabel("Mean force")

    if not (blim is None):
        ax1.set_ylim(blim[0], blim[1])
        ax3.set_ylim(blim[0], blim[1])
    if not (alim is None):
        ax2.set_ylim(alim[0], alim[1])
    if not (Flim is None):
        ax4.set_ylim(Flim[0], Flim[1])

    fig.suptitle(title)
    fig.savefig("{}drift_noise_compare{}.pdf".format(prefix, suffix))
    fig.savefig("{}drift_noise_compare{}.png".format(prefix, suffix))

if __name__ == "__main__":
    n_beads = 25
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)

    #msm_savedir = "msm_dih_dists"
    msm_savedir = "msm_dists"

    M = 1   # number of eigenvectors to use

    cg_savedir = "Ucg_eigenpair_1D"

    psi_hist = np.load(msm_savedir + "/psi1_n.npy")
    cv_r0 = np.load(msm_savedir + "/psi1_mid_bin.npy")
    cv_w = np.abs(cv_r0[1] - cv_r0[0])*np.ones(len(cv_r0), float)
    #cv_r0 = np.array([ [cv_r0[i]] for i in range(len(cv_r0)) ])
    #cv_r0 = cv_r0.reshape((len(cv_r0),1))

    print "creating Ucg..."
    # coarse-grain polymer potential with free parameters
    Ucg = iff.basis_library.OneDimensionalModel(1)
    Ucg.add_Gaussian_drift_basis(cv_r0, cv_w)
    Ucg.add_Gaussian_noise_basis(cv_r0, cv_w)
    Ucg.add_Gaussian_test_functions(cv_r0, cv_w)

    n_a = len(Ucg.a_funcs[1])
    n_b = len(Ucg.b_funcs[1])
    R = n_a + n_b           # number of free model parameters
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


    if not os.path.exists(cg_savedir + "/X.npy"):
        X = np.zeros((P, R), float)
        d = np.zeros(P, float)

        print "calculating matrix elements..."
        N_prev = 0 
        for n in range(len(traj_idxs)):
            print "traj: ", n+1
            sys.stdout.flush()
            idx1, idx2 = traj_idxs[n]
            Psi = np.load(msm_savedir + "/run_{}_{}_TIC_1.npy".format(idx1, idx2))
            N_curr = Psi.shape[0]

            # matrix elements 
            b1 = Ucg.evaluate_parametric_drift(Psi)
            a1 = Ucg.evaluate_parametric_noise(Psi)

            test_f = Ucg.test_functions(Psi)
            grad_f = Ucg.gradient_test_functions(Psi) 
            Lap_f = Ucg.laplacian_test_functions(Psi) 

            # partial sums for current traj
            curr_X1 = np.einsum("t,tr,tp->pr", Psi, b1, grad_f)
            curr_X2 = (-1/beta)*np.einsum("t,tr,tp->pr", Psi, a1, Lap_f)
            curr_d = kappa*np.einsum("t,tp->p", Psi, test_f)
             
            # recursive running average. Supposed to reduce floating point errors.
            X[:,:n_b] = (curr_X1 + N_prev*X[:,:n_b])/(N_prev + N_curr)
            X[:,n_b:] = (curr_X2 + N_prev*X[:,n_b:])/(N_prev + N_curr)
            d = (curr_d + N_prev*d)/(N_prev + N_curr)

            N_prev += N_curr

        if not os.path.exists(cg_savedir):
            os.mkdir(cg_savedir)
        os.chdir(cg_savedir)

        print "saving matrix..."
        np.save("X.npy", X)
        np.save("d.npy", d)

        with open("X_cond.dat", "w") as fout:
            fout.write(str(np.linalg.cond(X)))

        with open("Ntot.dat", "w") as fout:
            fout.write(str(N_prev))
    else:
        os.chdir(cg_savedir)
        X = np.load("X.npy")
        d = np.load("d.npy")

    ## regularize solution norm 
    #print "solving Ridge regression..."
    #alphas = np.logspace(-16, -2, 400)
    #alpha_star, coeff, all_soln, res_norm, reg_norm = iff.util.solve_ridge(alphas, X, d)

    #ylabel = "||\hat{c}||_2"
    #prefix = "ridge_"
    #suffix = "_{}".format(len(cv_r0))
    #title = "Ridge"

    #plot_regularization_soln(alphas, all_soln, res_norm, reg_norm, ylabel, title, prefix, suffix)

    ## regularize solution norm with preconditioning
    #print "solving Ridge regression with preconditioning..."
    #alphas = np.logspace(-16, -2, 400)
    #d1, d2, pre_X, pre_d = iff.util.Ruiz_preconditioner(X, d)
    #alpha_star, coeff, all_soln, res_norm, reg_norm = iff.util.solve_ridge(alphas, pre_X, pre_d)

    #ylabel = "||\hat{c}||_2"
    #prefix = "precond_ridge_"
    #suffix = "_{}".format(len(cv_r0))
    #title = "Precond Ridge"

    #plot_regularization_soln(alphas, all_soln, res_norm, reg_norm, ylabel, title, prefix, suffix)

    # regularize first derivative
    #print "solving regression. 1st deriv regularized..."
    #alphas = np.logspace(-6, 4, 400)
    #r = np.linspace(-1.1, 1.1, 200)
    #all_soln, res_norm, reg_norm = iff.util.solve_deriv_regularized(alphas, X, d, Ucg, r, order=1, variable_noise=True)

    #xlabel = r"TIC1 $\psi_1$"
    #ylabel = r"||\nabla b||_2 + ||\nabla a||_2"
    #prefix = "deriv_reg_"
    #suffix = "_{}".format(len(cv_r0))
    #title = "Regularize deriv"
    #plot_regularization_soln(alphas, all_soln, res_norm, reg_norm, ylabel, title, prefix, suffix)

    #alphas = [1e-4, 1, 1e4]
    #plot_select_drift_and_noise(alphas, X, d, Ucg, r, emp_r, emp_dF, xlabel, title, prefix, suffix, method="1st_deriv", alim=alim, blim=blim, Flim=Flim)

    r = np.linspace(min(cv_r0), max(cv_r0), 200)
    pmf = -np.log(psi_hist.astype(float))/beta
    pmf -= pmf.min()
    emp_dF = (pmf[1:] - pmf[:-1])/(cv_r0[1] - cv_r0[0])
    emp_r = 0.5*(cv_r0[1:] + cv_r0[:-1])

    Flim = (-50, 50)
    blim = (-0.001, 0.001)
    alim = (0, 0.0005)

    # regularize 2nd derivative
    alphas = np.logspace(-6, 3, 400)
    all_soln, res_norm, reg_norm = iff.util.solve_deriv_regularized(alphas, X, d, Ucg, r, order=1, variable_noise=True)

    xlabel = r"TIC1 $\psi_1$"
    ylabel = r"||\nabla b||_2 + ||\nabla a||_2"
    prefix = "deriv2_reg_"
    suffix = "_{}".format(len(cv_r0))
    title = "Regularize 2nd deriv"
    plot_regularization_soln(alphas, all_soln, res_norm, reg_norm, ylabel, title, prefix, suffix)

    alphas = [1e-10, 1e-4, 1]
    plot_select_drift_and_noise(alphas, X, d, Ucg, r, emp_r, emp_dF, xlabel, title, prefix, suffix, method="2nd_deriv", alim=alim, blim=blim, Flim=Flim)


    ###################################################
    # PLOT DRIFT, NOISE, AND MEAN FORCE. RIDGE
    ###################################################

    raise SystemExit

    alphas = [3e-4, 1e-7, 1e-13]
    xlabel = r"TIC1 $\psi_1$"
    prefix = "ridge_"
    suffix = "_{}".format(len(cv_r0))
    title = "Ridge"


    plot_select_drift_and_noise(alphas, X, d, Ucg, r, emp_r, emp_dF, xlabel, title, prefix, suffix, alim=alim, blim=blim, Flim=Flim)

    # On preconditioning: We determined that scaling the rows of X is like
    # reweighting the test functions. Since this has no physical justification
    # (it changes the meaning of our problem in a way that we don't care
    # about), we should not do row scaling. Column scaling is okay.

    # plot drift and diffusion with preconditioning (we determined that scaling
    # rows is reweighting test functions in a way that has no physical
    # justification)
    #alphas = [3e-4, 1e-7, 1e-13]
    #xlabel = r"TIC1 $\psi_1$"
    #prefix = "precond_ridge_"
    #suffix = "_{}".format(len(cv_r0))
    #title = "Precond Ridge"
    #blim = (-0.025, 0.025)
    #alim = (0, 0.0005)
    #plot_select_drift_and_noise(alphas, pre_X, pre_d, Ucg, r, emp_r, emp_dF, xlabel, title, prefix, suffix, right_precond=d2, alim=alim, blim=blim, Flim=Flim)

    raise SystemExit

    # PLAIN LEAST SQUARES
    coeff = lstsq_soln[0]
    b_coeff = coeff[:n_b]
    a_coeff = coeff[n_b:]

    #r = np.linspace(min(cv_r0[2:]), max(cv_r0[:-2]), 200)
    r = np.linspace(-1.1, 1.1, 200)

    drift, noise, d_noise, dF = calculate_drift_noise(r, Ucg, b_coeff, a_coeff)

    plt.figure()
    plt.plot(r, -dF)
    plt.xlabel(r"TIC1 $\psi_1$")
    plt.ylabel(r"Mean force $-\nabla W(\psi_1)$")
    plt.ylim(-100, 100)
    plt.savefig("grad_F_100.pdf")
    plt.savefig("grad_F_100.png")

    xmin, xmax = min(cv_r0), max(cv_r0)

    # PLOT PMF, DRIFT, AND NOISE
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 15))
    ax1.plot(cv_r0, pmf)
    ax1.set_ylabel(r"Traj PMF $-\log(P(\psi_1))$")
    ax1.set_ylim(0, 4)

    ax2.plot(r, drift)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylabel(r"Drift $b(\psi_1)$")

    ax3.plot(r, noise)
    ax3.set_xlim(xmin, xmax)
    ax3.set_xlabel(r"TIC1 $\psi_1$")
    ax3.set_ylabel(r"Noise $a(\psi_1)$")

    fig.savefig("drift_noise_1D_100.pdf")
    fig.savefig("drift_noise_1D_100.png")


    X_b = X[:,:n_b]
    X_b = np.ma.array(X_b, mask=(X_b == 0))

    X_a = X[:,n_b:]
    X_a = np.ma.array(X_a, mask=(X_a == 0))


    # PLOT MATRIX ELEMENTS
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
    fig.savefig("X_matrix_100.pdf")
    fig.savefig("X_matrix_100.png")


    plt.figure()
    plt.plot(d, 'ko')
    plt.xlabel(r"Test function $f_j$")
    plt.ylabel(r"$-\kappa_1\langle \psi_1, f_j\rangle$")
    plt.savefig("d_vector_100.pdf")
    plt.savefig("d_vector_100.png")

    # PLOT PRECONDITIONED MATRIX ELEMENTS
    X_b = pre_X[:,:n_b]
    X_b = np.ma.array(X_b, mask=(X_b == 0))

    X_a = pre_X[:,n_b:]
    X_a = np.ma.array(X_a, mask=(X_a == 0))


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
