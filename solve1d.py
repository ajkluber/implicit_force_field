from __future__ import print_function, absolute_import
import os
import sys
import time
import argparse
import glob
import numpy as np
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.use("Agg")
import matplotlib.pyplot as plt

import scipy.optimize

import implicit_force_field as iff
import implicit_force_field.loss_functions as loss


def eval_a(Ucg, xdata, coeff):
    a = np.zeros(xdata.shape[0], float)

    if len(Ucg.a_sym[0]) > 0:
        for i in range(len(Ucg.a_sym[0])):
            a += Ucg.a_funcs[0][i](xdata)

    for i in range(Ucg.n_noise_params):
        a += coeff[Ucg.n_pot_params + i]*Ucg.a_funcs[1][i](xdata)

    return a

def plot_U_a_solution(Ucg, coeff, psi_pmf, saveas):
    U_soln = Ucg.eval_U(xdata, coeff)
    U_soln -= U_soln.min()
    a_soln = eval_a(Ucg, xdata, coeff)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(xdata, U_soln, lw=2, label=r"EG soln")
    ax1.plot(xdata, psi_pmf, 'k--', lw=2, label="PMF")
    ax1.legend()
    ax1.set_xlabel(r"$\psi_2$")
    ax1.set_ylabel(r"$U(\psi_2)$ ($k_B$T)")

    ax2.plot(xdata, 1000*a_soln, lw=2)
    ax2.set_xlabel(r"$\psi_2$")
    ax2.set_ylabel(r"$a(\psi_2)$ x1000")
    fig.savefig(saveas + ".pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("slv_method", type=str)
    args = parser.parse_args()
    slv_method = args.slv_method

    T = 300.
    kb = 0.0083145
    beta = 1/(kb*T)
    msm_savedir = "msm_dists"
    n_basis_funcs = 40
    n_test_funcs = 100
    const_a = True
    fixed_a = True
    a_c = 0.005

    file_fixed_a_guess = "EG_coeff_fixed_a_guess.npy"
    file_const_a_guess = "EG_coeff_const_a_guess.npy"
    file_var_a_guess = "EG_coeff_var_a_{}_1.npy".format(slv_method)

    slv_opts = {"maxiter":1000,  "disp":True}
    #slv_opts = {"disp":True}
    #tol = 1e-10

    kappa = 1/np.load("tica_ti_ps.npy")[0]

    psinames = glob.glob("run_*_TIC_1.npy")
    #psi_trajs = [ np.concatenate([ np.load(x) for x in psinames ]) ]
    psi_trajs = [ np.load(x) for x in psinames ]

    temp_cv_r0 = np.load("psi1_mid_bin.npy")[1:-1]
    r0_basis = np.linspace(temp_cv_r0.min(), temp_cv_r0.max(), n_basis_funcs)
    w_basis = 2*np.abs(r0_basis[1] - r0_basis[0])*np.ones(len(r0_basis), float)
    r0_basis = r0_basis.reshape((-1, 1))

    r0_test = np.linspace(temp_cv_r0.min(), temp_cv_r0.max(), n_test_funcs)
    w_test = 2*np.abs(r0_test[1] - r0_test[0])*np.ones(len(r0_test), float)
    r0_test = r0_test.reshape((-1, 1))
    xdata = r0_test[:,0]

    print("creating models...")
    Ucg = iff.basis_library.OneDimensionalModel(1, beta, False, False)
    Ucg.add_linear_noise_term()
    Ucg.add_Gaussian_noise_basis(r0_basis, w_basis)

    Ucg.add_linear_potential()
    Ucg.add_Gaussian_potential_basis(r0_basis, w_basis)
    Ucg.add_Gaussian_test_functions(r0_test, w_test)

    Loss = loss.OneDimSpectralLoss(Ucg, kappa, psi_trajs, psi_trajs)

    c0_3 = np.load(file_var_a_guess)

    alpha_U = np.array([0])
    alpha_a = np.logspace(-10, 7, 30)

    #print("optimizing...")
    opt_soln = scipy.optimize.minimize(Loss.eval_loss, c0_3, method="CG", args=(0))
    all_coeffs, all_avg_cv, all_std_cv = Loss.solve(opt_soln.x, alpha_U, alpha_a)

    np.save("EG1d_coeffs.npy", all_coeffs)
    np.save("EG1d_avg_cv.npy", all_avg_cv)
    np.save("EG1d_std_cv.npy", all_std_cv)

    X, Y = np.meshgrid(alpha_U, alpha_a)

    all_coeffs = np.load("EG1d_coeffs.npy")
    all_avg_cv = np.load("EG1d_avg_cv.npy")
    all_std_cv = np.load("EG1d_std_cv.npy")

    #plt.figure()
    #plt.pcolormesh(X, Y, np.log10(all_avg_cv))
    #plt.semilogx()
    #plt.semilogy()
    #plt.colorbar()
    #plt.savefig("EG1d_cross_val_countour.pdf")

    plt.figure()
    plt.plot(alpha_a, all_avg_cv[0])
    plt.semilogx()
    plt.semilogy()
    plt.savefig("EG1d_cross_val_vs_alpha_a.pdf")

    #opt_coeff = np.copy(opt_soln.x)
    opt_coeff = all_coeffs[0,0]
    opt_coeff = all_coeffs[0,-10]
    opt_coeff[Loss.R_U:] = np.log(1 + np.exp(opt_coeff[Loss.R_U:]))

    #restart_coeff = opt_soln.x

    dx = xdata[1] - xdata[0]
    bin_edges = np.concatenate([ xdata - 0.5*dx, np.array([xdata[-1] + dx]) ])
    mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

    psi_n = np.zeros(len(bin_edges) - 1)

    for i in range(len(psi_trajs)):
        n, _ = np.histogram(psi_trajs[i], bins=bin_edges)
        psi_n += n

    Ntot = float(np.sum(psi_n))
    psi_n /= Ntot
    psi_pmf = -np.log(psi_n)
    psi_pmf -= psi_pmf.min()

    plot_U_a_solution(Ucg, opt_coeff, psi_pmf, "EG1d_crossval_U_var_a")

    raise SystemExit

    #all_coeffs = []
    #for i in range(len(alpha_U)):
    #    coeffs_1 = []
    #    for j in range(len(alpha_a)):
    #        coeff_sln, diff_cU, diff_ca = Loss._solve_general_a(c0_3, (alpha_U[i], alpha_a[j]), n_iters=10)
    #        coeffs_alpha.append(coeff_sln)        


    raise SystemExit

    # basis set with constant, but variable diffusion coefficient
    Ucg_const_a = iff.basis_library.OneDimensionalModel(1, beta, True, False)
    Ucg_const_a.add_linear_potential()
    Ucg_const_a.add_Gaussian_potential_basis(r0_basis, w_basis)
    Ucg_const_a.add_Gaussian_test_functions(r0_test, w_test)

    # basis set with fixed diffusion coefficient
    Ucg_fixa = iff.basis_library.OneDimensionalModel(1, beta, True, True, a_c=a_c)
    Ucg_fixa.add_linear_potential()
    Ucg_fixa.add_Gaussian_potential_basis(r0_basis, w_basis)
    Ucg_fixa.add_Gaussian_test_functions(r0_test, w_test)

    print("creating loss function...")
    Loss_const_a = loss.OneDimSpectralLoss(Ucg_const_a, kappa, psi_trajs, psi_trajs)
    Loss_fixa = loss.OneDimSpectralLoss(Ucg_fixa, kappa, psi_trajs, psi_trajs)

    # initial parameters
    c0 = np.ones(n_basis_funcs + 2, float) 
    c0 += 0.1*np.random.uniform(size=len(c0))
    if const_a:
        c0[-1] = a_c



    if not os.path.exists(file_const_a_guess):
        print("minimizing loss function with fixed a...")
        # with fixed diffusion coefficient the problem is linear
        coeff_1 = Loss_fixa.minimize_fixed_a_loss()
        np.save(file_fixed_a_guess, coeff_1)

        print("minimizing loss function with constant a...")
        # with constant, but variable diff coefficient, the problem is nonlinear.
        c0[:-1] = coeff_1
        result_2 = scipy.optimize.minimize(Loss_const_a.eval_loss, c0, method="Newton-CG", jac=Loss_const_a.eval_grad_loss, hess=Loss_const_a.eval_hess_loss, options=slv_opts)
        coeff_2 = result.x

        np.save(file_const_a_guess, coeff_2)
    else:
        coeff_1 = np.load(file_fixed_a_guess)
        coeff_2 = np.load(file_const_a_guess)

    Usln_1 = Ucg_fixa.eval_U(xdata, coeff_1)
    Usln_2 = Ucg_const_a.eval_U(xdata, coeff_2)

    #####################################################
    # POSITION-DEPENDENT DIFFUSION
    #####################################################
    #Precond = np.ones(Ucg.n_pot_params + Ucg.n_noise_params, float)
    #Precond[:Ucg.n_pot_params] = 0.0005
    #Precond[Ucg.n_pot_params:] = 0.25

    #iter = 1
    #vara_name = lambda meth, iter: "EG_coeff_var_a_{}_{}.npy".format(meth, iter)
    #while os.path.exists(

    # basis set with position-dependent diffusion coefficient
    Ucg = iff.basis_library.OneDimensionalModel(1, beta, False, False)
    Ucg.add_linear_noise_term()
    Ucg.add_Gaussian_noise_basis(r0_basis, w_basis)

    Ucg.add_linear_potential()
    Ucg.add_Gaussian_potential_basis(r0_basis, w_basis)
    Ucg.add_Gaussian_test_functions(r0_test, w_test)

    Loss = loss.OneDimSpectralLoss(Ucg, kappa, psi_trajs, psi_trajs)


    starttime = time.time()
    # with position-dependent diff coefficient, the problem is nonlinear.
    if os.path.exists(file_var_a_guess):
        c0_3 = np.load(file_var_a_guess)
    else:
        c0_3 = np.zeros(Ucg.n_pot_params + Ucg.n_noise_params, float) 
        c0_3[:Ucg.n_pot_params] = coeff_2[:-1]
        c0_3[Ucg.n_pot_params] = coeff_2[-1]
        c0_3[Ucg.n_pot_params + 1:] = 0.01*np.random.uniform(size=Ucg.n_noise_params - 1)

    print("minimizing loss function with variable a...")
    if slv_method in ["Nelder-Mead"]:
        result_3 = scipy.optimize.minimize(Loss.eval_loss, c0_3, method=slv_method, options=slv_opts)
    elif slv_method in ["CG", "L-BFGS-B"]:
        result_3 = scipy.optimize.minimize(Loss.eval_loss, c0_3, method=slv_method, jac=Loss.eval_grad_loss, options=slv_opts)
    elif slv_method in ["Newton-CG", "dogleg"]:
        result_3 = scipy.optimize.minimize(Loss.eval_loss, c0_3, method=slv_method, jac=Loss.eval_grad_loss, hess=Loss.eval_hess_loss, options=slv_opts)
    else:
        raise IOError("slv_method wrong!")


    sys.stdout.flush()
    stoptime = time.time()
    print("Method: {}       Took: {:.4f} min".format(slv_method, (stoptime - starttime)/60.))

    coeff_3 = result_3.x
    Usln_3 = Ucg.eval_U(xdata, coeff_3)

    np.save(file_var_a_guess, coeff_3)
    #coeff_3 = np.load("EG_coeff_var_a.npy")

    raise SystemExit

    plt.figure()
    plt.plot(xdata, Usln_1, lw=5, label=r"$a = {:.2e}$".format(a_c))
    plt.plot(xdata, Usln_2, lw=2, label=r"$a = {:.2e}$".format(coeff_2[-1]))
    plt.plot(xdata, Usln_3, lw=2, label=r"$a(\psi_2)$")
    plt.legend()
    plt.xlabel(r"$\psi_2$")
    plt.ylabel(r"$U$ ($k_BT$)")
    plt.savefig("EG_eff_pot.pdf")

    a_soln = eval_a(Ucg, xdata, coeff_3)

    plt.figure()
    plt.plot(xdata, a_soln, lw=2)
    plt.xlabel(r"$\psi_2$")
    plt.ylabel(r"$a(\psi_2)$")
    plt.savefig("EG_diff_coeff.pdf")
