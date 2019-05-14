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
    #file_var_a_guess = "EG_coeff_var_a_{}_1.npy".format(slv_method)

    slv_opts = {"maxiter":1000,  "disp":True}

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

    Ucg = iff.basis_library.OneDimensionalModel(1, beta, False, False)
    Ucg.add_constant_noise_term()
    Ucg.add_Gaussian_noise_basis(r0_basis, w_basis)

    Ucg.add_linear_potential()
    Ucg.add_Gaussian_potential_basis(r0_basis, w_basis)
    Ucg.add_Gaussian_test_functions(r0_test, w_test)

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


    all_coeffs = np.load("EG1d_coeffs.npy")
    alpha_U = np.load("EG1d_alpha_U.npy")
    alpha_a = np.load("EG1d_alpha_a.npy")
    all_avg_cv = np.load("EG1d_avg_cv.npy")
    all_std_cv = np.load("EG1d_std_cv.npy")


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i in range(len(all_coeffs)):
        for j in range(len(all_coeffs[i])):
            coeff = all_coeffs[i,j]
            coeff[Ucg.n_pot_params:] = np.log(1 + np.exp(coeff[Ucg.n_pot_params:]))

            U_soln = Ucg.eval_U(xdata, coeff)
            U_soln -= U_soln.min()
            a_soln = eval_a(Ucg, xdata, coeff)

            ax1.plot(xdata, U_soln, lw=2)
            ax2.plot(xdata, 1000*a_soln, lw=2)

    saveas = "test_EG1d_all_solns"

    ax1.plot(xdata, psi_pmf, 'k--', lw=2, label="PMF")
    ax1.legend()
    ax1.set_ylim(0, 10)
    ax1.set_xlabel(r"$\psi_2$")
    ax1.set_ylabel(r"$U(\psi_2)$ ($k_B$T)")

    ax2.set_xlabel(r"$\psi_2$")
    ax2.set_ylabel(r"$a(\psi_2)$ x1000")
    ax2.semilogy()
    fig.savefig(saveas + ".pdf")
