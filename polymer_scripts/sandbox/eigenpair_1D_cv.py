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

import mdtraj as md

import implicit_force_field as iff


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

def plot_select_Ucg_CV(alphas, A, b, Ucg, r, emp_r, emp_pmf, xlabel, title,
        prefix, suffix, method="ridge", right_precond=None, Ulim=None):

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

    alpha_star, coeff, all_soln, res_norm, reg_norm = iff.util.solve_ridge(alphas, A, b, right_precond=right_precond)

    ax1.plot(emp_r, emp_pmf, 'k', lw=2, label="PMF")
    for i in range(len(alphas)):
        # evaluate drift
        c_coeff = all_soln[i][:-1]
        A = all_soln[i][-1]*np.ones(len(r), float)

        U_r = np.zeros(len(r), float)
        for k in range(len(c_coeff)):
            U_r += c_coeff[k]*Ucg.cv_U_funcs[k](r)

        ax1.plot(r, U_r, label=r"$\alpha = {:.2e}$".format(alphas[i]))

    ax1.legend(loc=1, fancybox=False, edgecolor="k", facecolor="w", framealpha=1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Potential")

    if not (Ulim is None):
        ax1.set_ylim(Ulim[0], Ulim[1])

    fig.suptitle(title)
    fig.savefig("{}Ucg_cv{}.pdf".format(prefix, suffix))
    fig.savefig("{}Ucg_cv{}.png".format(prefix, suffix))

if __name__ == "__main__":
    n_beads = 25
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)

    #msm_savedir = "msm_dih_dists"
    msm_savedir = "msm_dists"

    M = 1   # number of eigenvectors to use

    cg_savedir = "Ucg_CV_eigenpair_1D"

    ply_idxs = np.arange(25)
    pair_idxs = []
    for i in range(len(ply_idxs) - 1):
        for j in range(i + 4, len(ply_idxs)):
            pair_idxs.append([ply_idxs[i], ply_idxs[j]])
    pair_idxs = np.array(pair_idxs)

    psi_hist = np.load(msm_savedir + "/psi1_n.npy")
    cv_r0 = np.load(msm_savedir + "/psi1_mid_bin.npy")
    cv_w = np.abs(cv_r0[1] - cv_r0[0])*np.ones(len(cv_r0), float)
    cv_r0 = cv_r0.reshape((len(cv_r0),1))
    cv_coeff = np.load(msm_savedir + "/tica_eigenvects.npy")[:,:M]
    cv_mean = np.load(msm_savedir + "/tica_mean.npy")

    print "creating Ucg..."
    # coarse-grain polymer potential with free parameters
    Ucg = iff.basis_library.PolymerModel(n_beads)
    Ucg.define_collective_variables(["dist"], pair_idxs, cv_coeff, cv_mean)
    Ucg.collective_variable_test_funcs(cv_r0, cv_w)
    Ucg._add_Gaussian_cv_potentials(cv_r0, cv_w)

    R = len(Ucg.cv_U_funcs)    # number of free model parameters
    P = Ucg.n_test_funcs_cv    # number of test functions

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
    kappa = np.array([kappa])

    if not os.path.exists(cg_savedir + "/X.npy"):
        X = np.zeros((P, R + 1), float)
        d = np.zeros(P, float)

        print "calculating matrix elements..."
        N_prev = 0 
        for n in range(len(traj_idxs)):
            print "traj: ", n+1
            sys.stdout.flush()
            idx1, idx2 = traj_idxs[n]
            psi_traj = np.load(msm_savedir + "/run_{}_{}_TIC_1.npy".format(idx1, idx2)).reshape(-1,1)

            start_idx = 0
            chunk_num = 1
            for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
                print "    chunk: ", chunk_num
                sys.stdout.flush()
                chunk_num += 1
                N_curr = chunk.n_frames

                Psi = psi_traj[start_idx:start_idx + N_curr,:]

                # calculate gradient of fixed and parametric potential terms
                grad_U0 = Ucg.gradient_U0(chunk)
                grad_U1 = Ucg.gradient_U1_cv(chunk, Psi)

                # calculate test function values, gradient, and Laplacian
                test_f = Ucg.test_functions_cv(Psi)
                grad_f, Lap_f = Ucg.gradient_and_laplacian_test_functions_cv(chunk, Psi) 

                # very useful einstein summation function to calculate
                # dot products with eigenvectors
                curr_X1 = np.einsum("tm,tdr,tdp->mpr", Psi, -grad_U1, grad_f).reshape((M*P, R))
                curr_X2 = np.einsum("m,tm,tp->mp", kappa, Psi, test_f).reshape(M*P)

                curr_d1 = np.einsum("tm,td,tdp->mp", Psi, grad_U0, grad_f).reshape(M*P)
                curr_d2 = (-1./beta)*np.einsum("tm,tp->mp", Psi, Lap_f).reshape(M*P)
                curr_d = curr_d1 + curr_d2

                X[:,:-1] = (curr_X1 + N_prev*X[:,:-1])/(N_prev + N_curr)
                X[:,-1] = (curr_X2 + N_prev*X[:,-1])/(N_prev + N_curr)
                d = (curr_d + N_prev*d)/(N_prev + N_curr)

                start_idx += N_curr
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


    ###################################################
    # PLOT PMF and Ucg
    ###################################################
    r = np.linspace(min(cv_r0), max(cv_r0), 200)
    pmf = -np.log(psi_hist.astype(float))/beta
    pmf -= pmf.min()
    emp_dF = (pmf[1:] - pmf[:-1])/(cv_r0[1] - cv_r0[0])
    emp_r = 0.5*(cv_r0[1:] + cv_r0[:-1])

    Ulim = (-10, 10)
    #Ulim = (-100, 100)
    alim = (0, 0.0005)

    # regularize 2nd derivative
    print "plotting regularized solutions..."
    blim = (-0.01, 0.01)
    xlabel = r"TIC1 $\psi_1$"
    suffix = "_{}".format(len(cv_r0))

    alphas = np.logspace(-7, 1, 400)
    select_alphas = [1e-4, 1, 1e2]
    ylabel = r"||c||^2_2"
    prefix = "ridge_U_CV_"
    title = "Ridge"
    reg_method = "ridge"
    alpha_star, coeff, all_soln, res_norm, reg_norm = iff.util.solve_ridge(alphas, X, d)

    plot_regularization_soln(alphas, all_soln, res_norm, reg_norm, ylabel, title, prefix, suffix)

    plot_select_Ucg_CV(select_alphas, X, d, Ucg, r, cv_r0, pmf, 
            xlabel, title, prefix, suffix, method=reg_method, Ulim=Ulim)

