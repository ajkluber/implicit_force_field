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

def junk():
    pass
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

def plot_cv_score():
    cv_score = np.array(cv_score)
    idx = np.argwhere(cv_score <= 1.10*cv_score.min())[0,0]
    alpha_star = alphas[idx]

    plt.figure()
    plt.plot(alphas,  cv_score)
    plt.axvline(alpha_star, ls='--', color="k")
    plt.semilogx(True)
    plt.semilogy(True)
    plt.title(r"$\alpha^*={:.4e}$".format(alpha_star))
    plt.xlabel(r"Regularization $\alpha$")
    plt.ylabel("MSE on test data")
    #plt.savefig("D2_penalty_cross_val.pdf")
    #plt.savefig("D2_penalty_cross_val.png")
    plt.savefig("ridgecv_cross_val.pdf")
    plt.savefig("ridgecv_cross_val.png")
    plt.show()


def find_ratio_of_norms(X, d, Ucg, r):
    """ """
    soln = iff.util.solve_deriv_regularized([1e-14], X, d, Ucg, r, order=2, variable_noise=True)[0][0]
    n_b = len(Ucg.b_funcs[1])
    drift, noise, d_noise, dF = calculate_drift_noise(r, Ucg, soln[:n_b], soln[n_b:])

    D2 = iff.util.D2_operator(Ucg, r, variable_noise=True)

    d2_drift = np.dot(D2[:,:n_b], soln[:n_b])
    d2_noise = np.dot(D2[:,n_b:], soln[n_b:])

    norm_d2_b = np.linalg.norm(d2_drift)
    norm_d2_a = np.linalg.norm(d2_noise)

    ratio_norms = (norm_d2_b/norm_d2_a)**2

    return ratio_norms

def Gauss(x, r0, w):
    return np.exp(-0.5*((r0 - x)/w)**2)

def get_smooth_potential_mean_force(cv_r0, cv_w, pmf, alpha_star, scan_alphas=True):
    G = np.array([ Gauss(cv_r0, cv_r0[i], cv_w[i]) for i in range(len(cv_r0)) ]).T

    D2 = np.zeros(G.shape, float)
    for i in range(len(cv_r0)):
        y = Gauss(cv_r0, cv_r0[i], cv_w[i])

        # centered differences
        D2[1:-1,i] = (y[2:] - 2*y[1:-1] + y[:-2])

        # forward and backward difference
        D2[0,i] = (y[0] - 2*y[1] + y[2])
        D2[-1,i] = (y[-1] - 2*y[-2] + y[-3])
    D2 /= (cv_r0[1] - cv_r0[0])**2

    if scan_alphas:
        alphas = np.logspace(-8, -4, 500)
        all_soln = []
        res_norm = []
        reg_norm = []
        for i in range(len(alphas)):
            # regularize the second derivative of solution
            A_reg = np.dot(G.T, G) + alphas[i]*np.dot(D2.T, D2)
            b_reg = np.dot(G.T, pmf)

            x = np.linalg.lstsq(A_reg, b_reg, rcond=1e-11)[0]

            all_soln.append(x)
            res_norm.append(np.linalg.norm(np.dot(G, x) - pmf))
            reg_norm.append(np.linalg.norm(np.dot(D2, x)))

        plot_regularization_soln(alphas, all_soln, res_norm, reg_norm, r"||\frac{d^2 F}{dx^2}||_2", "Smooth PMF", "pmf_smooth_", "_100")

    # interpolate pmf with smooth function
    emp_dF = (pmf[1:] - pmf[:-1])/(cv_r0[1] - cv_r0[0])
    emp_r = 0.5*(cv_r0[1:] + cv_r0[:-1])
    r = np.linspace(np.min(cv_r0), np.max(cv_r0), 500)

    alphas = [1e-8, 1e-7, 1e-6, 1e-5]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax3.axhline(0, c='k', lw=1)
    for i in range(len(alphas)):
        # regularize the second derivative of solution
        A_reg = np.dot(G.T, G) + alphas[i]*np.dot(D2.T, D2)
        b_reg = np.dot(G.T, pmf)

        pmf_ck = np.linalg.lstsq(A_reg, b_reg, rcond=1e-11)[0]

        # plot interpolated pmf
        int_pmf = np.sum([ pmf_ck[n]*Gauss(r, cv_r0[n], cv_w[n]) for n in range(len(cv_r0)) ], axis=0)
        ax1.plot(r, int_pmf, label=r"$\alpha = {:.0e}$".format(alphas[i]))

        # zoom in on barrier region
        ax2.plot(r, int_pmf)

        # plot interpolated mean force
        int_dF = (int_pmf[1:] - int_pmf[:-1])/(r[1] - r[0])
        int_r = 0.5*(r[1:] + r[:-1])
        ax3.plot(int_r, -int_dF)

    # plot empirical free energy and mean force
    ax1.plot(cv_r0, pmf, 'k--', label="Empirical", lw=2)
    ax2.plot(cv_r0, pmf, 'k--')
    ax3.plot(emp_r, -emp_dF, 'k--')

    ax1.set_ylim(-.5, 4)
    ax2.set_ylim(3, 3.6)
    ax2.set_xlim(-.5, .5)
    ax3.set_ylim(-13, 13)

    ax1.legend(loc=4, framealpha=1, fancybox=False, edgecolor="k", facecolor="w")
    ax1.set_xlabel(r"TIC1 $\psi_1$")
    ax2.set_xlabel(r"TIC1 $\psi_1$")
    ax3.set_xlabel(r"TIC1 $\psi_1$")

    ax1.set_title("Free energy")
    ax2.set_title("Free energy")
    ax3.set_title("Mean Force")

    fig.savefig("pmf_smooth_samples.pdf")
    fig.savefig("pmf_smooth_samples.png")

    #lalpha_star = 1e-6
    A_reg = np.dot(G.T, G) + alpha_star*np.dot(D2.T, D2)
    b_reg = np.dot(G.T, pmf)
    pmf_ck = np.linalg.lstsq(A_reg, b_reg, rcond=1e-11)[0]

    return pmf_ck

def calculate_drift_noise(r, Ucg, b_coeff, a_coeff):

    drift = np.zeros(len(r), float)
    for i in range(len(b_coeff)):
        drift += b_coeff[i]*Ucg.b_funcs[1][i](r)

    if len(a_coeff) == 1:
        noise = a_coeff[0]*np.ones(len(r), float)
        d_noise = np.zeros(len(r), float)
        dF = -drift/a_coeff[0]
    else:
        noise = np.zeros(len(r), float)
        for i in range(len(a_coeff)):
            noise += a_coeff[i]*Ucg.a_funcs[1][i](r)

        d_noise = np.zeros(len(r), float)
        for i in range(len(a_coeff)):
            d_noise += a_coeff[i]*sympy.lambdify(x_sym, Ucg.a_sym[1][i].diff(x_sym), modules="numpy")(r)

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
        title, prefix, suffix, method="ridge", weight_a=1, right_precond=None,
        alim=None, blim=None, Flim=None, variable_noise=True, D2=None):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))

    n_b = len(Ucg.b_funcs[1])
    n_a = len(Ucg.a_funcs[1])

    ax4.plot(emp_r, -emp_dF, 'k', label="Emp")
    ax4.axhline(0, c='k')

    if method == "ridge":
        alpha_star, coeff, all_soln, res_norm, reg_norm = iff.util.solve_ridge(alphas, A, b, right_precond=right_precond)
    elif method == "1st_deriv":
        all_soln, res_norm, reg_norm = iff.util.solve_deriv_regularized(alphas, A, b, Ucg, r, weight_a=1, order=1, variable_noise=variable_noise)
    elif method == "2nd_deriv":
        all_soln, res_norm, reg_norm = iff.util.solve_deriv_regularized(alphas, A, b, Ucg, r, weight_a=1, order=2, variable_noise=variable_noise)
    elif method == "D2":
        all_soln, res_norm, reg_norm, cv_score = iff.util.solve_D2_regularized(alphas, A, b, D2, n_b=n_b, weight_a=weight_a)

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

    ax1.legend(loc=1, fancybox=False, edgecolor="k", facecolor="w", framealpha=1)
    ax4.legend(loc=1, fancybox=False, edgecolor="k", facecolor="w", framealpha=1)

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

    #constD = True
    constD = False
    reg_method = "ridge"
    # reg_method = "D2"

    if constD:
        cg_savedir = "Ucg_eigenpair_1D_constD"
    else:
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
    Ucg.add_Gaussian_test_functions(cv_r0, cv_w)
    n_b = len(Ucg.b_funcs[1])

    if not constD:
        Ucg.add_Gaussian_noise_basis(cv_r0, cv_w)
        n_a = len(Ucg.a_funcs[1])
    else:
        n_a = 1

    R = n_b + n_a           # number of free model parameters
    P = len(Ucg.f_funcs)    # number of test functions

    ##########################################################
    # EIGENPAIR MATRIX ELEMENTS. VARIABLE DIFFUSION COEFF
    ##########################################################
    #topfile = glob.glob("run_{}/".format(run_idx) + name + "_min_cent.pdb")[0]
    #trajnames = glob.glob("run_{}/".format(run_idx) + name + "_traj_cent_*.dcd")
    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd")
    xmlnames = glob.glob("run_*/" + name + "_final_state_*.xml")

    traj_idxs = []
    if len(trajnames) == 0:
        temp_filenames = xmlnames
        notrajs = True
    else:
        temp_filenames = trajnames
        notrajs = False

    for i in range(len(temp_filenames)):
        tname = temp_filenames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        traj_idxs.append([idx1, idx2])

    kappa = 1./np.load(msm_savedir + "/tica_ti.npy")[0]

    if not os.path.exists(cg_savedir + "/X.npy"):
        # penalty on the second derivative
        if notrajs:
            raise ValueError("Check that trajectory files exist.")

        D2 = np.zeros((R, R), float)

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

            # matrix elements calculated using a recursive running average to
            # reduce floating point errors.
            b1 = Ucg.evaluate_parametric_drift(Psi)

            test_f = Ucg.test_functions(Psi)
            grad_f = Ucg.gradient_test_functions(Psi)
            Lap_f = Ucg.laplacian_test_functions(Psi)

            curr_D2 = Ucg.evaluate_D2_matrix(Psi)

            curr_X1 = np.einsum("t,tr,tp->pr", Psi, b1, grad_f)
            curr_d = -kappa*np.einsum("t,tp->p", Psi, test_f)

            X[:,:n_b] = (curr_X1 + N_prev*X[:,:n_b])/(N_prev + N_curr)
            D2 = (curr_D2 + N_prev*D2)/(N_prev + N_curr)

            if constD:
                curr_X2 = np.einsum("t,tp->p", Psi, Lap_f)/beta
                X[:,-1] = (curr_X2 + N_prev*X[:,-1])/(N_prev + N_curr)

            else:
                # Is this negative sign is wrong??
                a1 = Ucg.evaluate_parametric_noise(Psi)
                curr_X2 = np.einsum("t,tr,tp->pr", Psi, a1, Lap_f)/beta
                X[:,n_b:] = (curr_X2 + N_prev*X[:,n_b:])/(N_prev + N_curr)
            d = (curr_d + N_prev*d)/(N_prev + N_curr)

            N_prev += N_curr

        if not os.path.exists(cg_savedir):
            os.mkdir(cg_savedir)
        os.chdir(cg_savedir)

        print "saving matrix X and d..."
        np.save("D2.npy", D2)
        np.save("X.npy", X)
        np.save("d.npy", d)

        with open("X_cond.dat", "w") as fout:
            fout.write(str(np.linalg.cond(X)))

        with open("Ntot.dat", "w") as fout:
            fout.write(str(N_prev))
    else:
        print "loading X and d..."
        os.chdir(cg_savedir)
        D2 = np.load("D2.npy")
        X = np.load("X.npy")
        d = np.load("d.npy")

    ratio_norms = np.sqrt(1.238697)  # found using alpha=1e-14

    r = np.linspace(min(cv_r0), max(cv_r0), 200)
    pmf = -np.log(psi_hist.astype(float))/beta
    pmf -= pmf.min()
    emp_dF = (pmf[1:] - pmf[:-1])/(cv_r0[1] - cv_r0[0])
    emp_r = 0.5*(cv_r0[1:] + cv_r0[:-1])

    ###################################################
    # PLOT DRIFT, NOISE, AND MEAN FORCE. RIDGE
    ###################################################

    # interpolate pmf with smooth function
    alpha_star = 5e-5
    pmf_ck = get_smooth_potential_mean_force(cv_r0, cv_w, pmf, alpha_star, scan_alphas=False)

    int_pmf = np.sum([ pmf_ck[i]*Gauss(r, cv_r0[i], cv_w[i]) for i in range(len(cv_r0)) ], axis=0)
    int_dF = (int_pmf[1:] - int_pmf[:-1])/(r[1] - r[0])
    int_r = 0.5*(r[1:] + r[:-1])

    #Flim = (-50, 50)
    Flim = (-10, 10)
    alim = (0, 0.0005)

    # raise SystemExit

    # regularize 2nd derivative
    print "plotting regularized solutions..."
    if constD:
        blim = (-0.01, 0.01)
        xlabel = r"TIC1 $\psi_1$"
        suffix = "_{}".format(len(cv_r0))

        if reg_method == "2nd_deriv":
            alphas = np.logspace(-14, -9, 400)
            select_alphas = [1e-16, 1e-14, 1e-12]
            ylabel = r"||\frac{d^2 b}{dx^2}||^2_2"
            prefix = "deriv2_reg_constD_"
            title = "Regularize 2nd deriv"
            all_soln, res_norm, reg_norm = iff.util.solve_deriv_regularized(alphas, X, d, Ucg, r, order=2, variable_noise=False)

        elif reg_method == "ridge":
            alphas = np.logspace(-15, -10, 400)
            select_alphas = [1e-10, 1e-7, 1e-4]
            ylabel = r"||c||^2_2"
            prefix = "ridge_constD_"
            title = "Ridge"
            alpha_star, coeff, all_soln, res_norm, reg_norm = iff.util.solve_ridge(alphas, X, d)


        plot_regularization_soln(alphas, all_soln, res_norm, reg_norm, ylabel, title, prefix, suffix)

        plot_select_drift_and_noise(select_alphas, X, d, Ucg, r, int_r, int_dF,
                xlabel, title, prefix, suffix, method=reg_method,
                weight_a=ratio_norms, alim=alim, blim=blim, Flim=Flim,
                variable_noise=False, D2=D2)
    else:
        blim = (-0.001, 0.001)
        xlabel = r"TIC1 $\psi_1$"
        suffix = "_{}".format(len(cv_r0))

        if reg_method == "2nd_deriv":
            alphas = np.logspace(-14, -9, 400)
            #select_alphas = [1e-14, 1e-12, 1e-10]
            select_alphas = [1e-16, 1e-10, 5e-4]
            suffix = "_over_reg_example"
            ylabel = r"||\frac{d^2 b}{dx^2}||^2_2 + ||\frac{d^2 a}{dx^2}||^2_2"
            prefix = "deriv2_reg_"
            title = "Regularize 2nd deriv"
            all_soln, res_norm, reg_norm = iff.util.solve_deriv_regularized(alphas, X, d, Ucg, r, order=2, variable_noise=True)
        elif reg_method == "ridge":
            alphas = np.logspace(-15, -10, 400)
            select_alphas = [1e-10, 1e-7, 1e-4]
            #select_alphas = [1e-5, 1e-3, 1e-1]
            ylabel = r"||c||^2_2"
            prefix = "ridge_"
            title = "Ridge"
            alpha_star, coeff, all_soln, res_norm, reg_norm, cv_score = iff.util.solve_ridge(alphas, X, d)
        elif reg_method == "D2":
            alphas = np.logspace(-20, -6, 500)
            #select_alphas = [1e-12, 1e-10, 1e-8]
            ylabel = r"||D2||^2_2"
            prefix = "D2_"
            title = "D2"
            #alpha_star, coeff, all_soln, res_norm, reg_norm = iff.util.solve_D2_regularized(alphas, X, d)
            all_soln, res_norm, reg_norm, cv_score = iff.util.solve_D2_regularized(alphas, X, d, D2, n_b=100, weight_a=1)

            idx = np.argwhere(cv_score <= 1.10*cv_score.min())[0,0]
            alpha_star = alphas[idx]
            coeff_star = np.linalg.lstsq(np.dot(X.T, X) + alpha_star*D2, np.dot(X.T, d), rcond=1e-11)[0]
            b_coeff, a_coeff = coeff_star[:n_b], coeff_star[n_b:]
            select_alphas = [1e-11, 1e-7, alpha_star, 1e-3]
            # select_alphas = [1e-11, 1e-7, 1e-3]

        plot_regularization_soln(alphas, all_soln, res_norm, reg_norm, ylabel, title, prefix, suffix)

        # plot a couple solutions
        plot_select_drift_and_noise(select_alphas, X, d, Ucg, r, emp_r, emp_dF,
                xlabel, title, prefix, suffix, method=reg_method, weight_a=1,
                alim=alim, blim=blim, Flim=Flim, variable_noise=True, D2=D2)


    raise SystemExit

    alphas = [3e-4, 1e-7, 1e-13]
    xlabel = r"TIC1 $\psi_1$"
    prefix = "ridge_"
    suffix = "_{}".format(len(cv_r0))
    title = "Ridge"



    plot_select_drift_and_noise(alphas, X, d, Ucg, r, int_r, int_dF, xlabel, title, prefix, suffix, alim=alim, blim=blim, Flim=Flim)

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
    #plot_select_drift_and_noise(alphas, pre_X, pre_d, Ucg, r, int_r, int_dF, xlabel, title, prefix, suffix, right_precond=d2, alim=alim, blim=blim, Flim=Flim)

    raise SystemExit

    from scipy.optimize import nnls

    soln, res_norm = nnls(X, d)

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
