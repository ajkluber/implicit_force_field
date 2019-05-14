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

    kappa = 1/np.load("tica_ti_ps.npy")[0]

    psinames = glob.glob("run_*_TIC_1.npy")
    psi_trajs = [ np.concatenate([ np.load(x) for x in psinames ][:10]) ]
    #psi_trajs = [ np.load(x) for x in psinames ]

    temp_cv_r0 = np.load("psi1_mid_bin.npy")[1:-1]
    r0_basis = np.linspace(temp_cv_r0.min(), temp_cv_r0.max(), n_basis_funcs)
    w_basis = 2*np.abs(r0_basis[1] - r0_basis[0])*np.ones(len(r0_basis), float)
    r0_basis = r0_basis.reshape((-1, 1))

    r0_test = np.linspace(temp_cv_r0.min(), temp_cv_r0.max(), n_test_funcs)
    w_test = 2*np.abs(r0_test[1] - r0_test[0])*np.ones(len(r0_test), float)
    r0_test = r0_test.reshape((-1, 1))

    print("creating models...")

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


    # initial parameters
    c0 = np.ones(n_basis_funcs + 2, float) 
    c0 += 0.1*np.random.uniform(size=len(c0))
    if const_a:
        c0[-1] = a_c

    slv_opts = {"maxiter":100,  "disp":True}
    #slv_opts = {"disp":True}
    #tol = 1e-10
    xdata = r0_test[:,0]

    coeff_fixed_a_guess = "EG_coeff_fixed_a_guess.npy"
    coeff_const_a_guess = "EG_coeff_const_a_guess.npy"

    coeff_1 = np.load(coeff_fixed_a_guess)
    coeff_2 = np.load(coeff_const_a_guess)

    Usln_1 = Ucg_fixa.eval_U(xdata, coeff_1)
    Usln_2 = Ucg_const_a.eval_U(xdata, coeff_2)

    # basis set with position-dependent diffusion coefficient
    Ucg = iff.basis_library.OneDimensionalModel(1, beta, False, False)
    Ucg.add_linear_noise_term()
    Ucg.add_Gaussian_noise_basis(r0_basis, w_basis)

    Ucg.add_linear_potential()
    Ucg.add_Gaussian_potential_basis(r0_basis, w_basis)
    Ucg.add_Gaussian_test_functions(r0_test, w_test)

    coeff_3 = np.load("EG_coeff_var_a_{}.npy".format(slv_method))
    Usln_3 = Ucg.eval_U(xdata, coeff_3)

    print("plotting...")
    plt.figure()
    plt.plot(xdata, Usln_1, lw=5, label=r"$a = {:.2e}$".format(a_c))
    plt.plot(xdata, Usln_2, lw=2, label=r"$a = {:.2e}$".format(coeff_2[-1]))
    plt.plot(xdata, Usln_3, lw=2, label=r"$a(\psi_2)$")
    plt.legend()
    plt.xlabel(r"$\psi_2$")
    plt.ylabel(r"$U$ ($k_BT$)")
    plt.savefig("EG_eff_pot_{}.pdf".format(slv_method))

    a_soln = eval_a(Ucg, xdata, coeff_3)

    plt.figure()
    plt.plot(xdata, a_soln, lw=2)
    plt.xlabel(r"$\psi_2$")
    plt.ylabel(r"$a(\psi_2)$")
    plt.savefig("EG_diff_coeff_{}.pdf".format(slv_method))
