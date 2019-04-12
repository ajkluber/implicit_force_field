from __future__ import print_function, absolute_import
import os
import glob
import numpy as np
#import matplotlib as mpl
#mpl.rcParams['mathtext.fontset'] = 'cm'
#mpl.rcParams['mathtext.rm'] = 'serif'
#mpl.use("Agg")
#import matplotlib.pyplot as plt

import scipy.optimize

import implicit_force_field as iff
import implicit_force_field.loss_functions as loss

if __name__ == "__main__":
    T = 300.
    kb = 0.0083145
    beta = 1/(kb*T)
    msm_savedir = "msm_dists"
    n_basis_funcs = 40
    n_test_funcs = 100

    os.chdir(msm_savedir)

    # create one-dimensional model. Gaussian basis functions for potential and
    # noise function
    Ucg = iff.basis_library.OneDimensionalModel(1, beta)

    temp_cv_r0 = np.load("psi1_mid_bin.npy")[1:-1]

    r0_basis = np.linspace(temp_cv_r0.min(), temp_cv_r0.max(), n_basis_funcs)
    w_basis = 2*np.abs(r0_basis[1] - r0_basis[0])*np.ones(len(r0_basis), float)
    r0_basis = cv_r0.reshape((-1, 1))
    Ucg.add_Gaussian_potential_basis(r0_basis, w_basis)
    Ucg.add_Gaussian_noise_basis(r0_basis, w_basis)

    r0_test = np.linspace(temp_cv_r0.min(), temp_cv_r0.max(), n_test_funcs)
    w_test = 2*np.abs(r0_test[1] - r0_test[0])*np.ones(len(r0_test), float)
    r0_test = cv_r0.reshape((-1, 1))
    Ucg.add_Gaussian_test_functions(r0_test, w_test)

    R_a = Ucg.n_pot_params
    R_u = Ucg.n_noise_params
    c0 = np.ones(R_a + R_u, float)

    kappa = 1/np.load("tica_ti_ps.npy")[0]

    psinames = glob.glob("run_*_TIC_1.npy")
    psi_trajs = [ np.load(x) for x in psiname ]

    loss_func = loss_functions.OneDimSpectralLoss(Ucg, kappa, psi_trajs, psi_trajs)

    result = scipy.optimize.minimize(loss_func.eval_loss, c0)
