from __future__ import print_function, absolute_import
import numpy as np
import os

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.use("Agg")
import matplotlib.pyplot as plt

import implicit_force_field.polymer_scripts.util as util

if __name__ == "__main__":
    using_U0 = True
    using_cv = True
    using_D2 = False
    fix_back = True
    fix_exvol = False
    a_coeff = True
    n_basis = 40
    n_test = 100
    n_pair_gauss = None
    M = 1
    n_beads = 25
    msm_savedir = "msm_dists"
    beta = 1./(0.0083145*300)

    method_code = ["EG", "EG", "FM", "FM"]
    cg_method = ["eigenpair", "eigenpair", "force matching", "force matching"]
    bondcut = [4, 3, 4, 3]
    linesty = ["-", "--", "-", "--"]
    colors = ["k", "k", "r", "r"]
    legend = ["spectral", "spectral", "force-matching", "force-matching"] 

    #plt.figure()
    #for i in range(len(method_code)):

    #    cg_savedir = "Ucg_{}_fixback_CV_1_40_100_bondcut_{}".format(method_code[i], bondcut[i])
    #    if method_code[i] == "EG":
    #        cg_savedir += "_fixed_a"
    #    os.chdir(cg_savedir)
    #    if os.path.exists("alpha_sigma_valid_mse.npy"):
    #        label = "{}  excl $|i - j| < {}$".format(legend[i], bondcut[i])
    #        vl_mse = np.load("alpha_sigma_valid_mse.npy")
    #        sigma_idx, alpha_idx = np.load("best_sigma_alpha_idx.npy")
    #        new_sigma = np.load("scaled_sigma_vals.npy")
    #        at_best_alpha = vl_mse[:, alpha_idx, 0]
    #        plt.plot(new_sigma, at_best_alpha, color=colors[i], ls=linesty[i], label=label)
    #    os.chdir("..")

    #plt.legend()
    #plt.semilogy()
    #plt.xlabel(r"Scaled radius $\sigma'$ (nm)")
    #plt.ylabel(r"Crossval score")

    #plt.savefig("plots/compare_crossval_vs_scaled_sigma.pdf")
    #plt.savefig("plots/compare_crossval_vs_scaled_sigma.png")

    plt.figure()
    for i in range(len(method_code)):
        label = "{}  excl $|i - j| < {}$".format(legend[i], bondcut[i])
        print(label)

        cg_savedir = util.Ucg_dirname(cg_method[i], M, using_U0, fix_back,
                fix_exvol, bondcut[i], using_cv, n_cv_basis_funcs=n_basis,
                n_cv_test_funcs=n_test, a_coeff=a_coeff)

        if os.path.exists(cg_savedir + "/rdg_fixed_sigma_cstar.npy"):
            Ucg, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(msm_savedir,
                    n_beads, M, beta, fix_back, fix_exvol, using_cv, using_D2,
                    n_basis, n_test, n_pair_gauss, bondcut[i], a_coeff=True)

            cv_grid = np.linspace(1.3*cv_r0_basis.min(), 1.2*cv_r0_basis.max(), 200)

            coeff = np.load(cg_savedir + "/rdg_fixed_sigma_cstar.npy")
            Ucv = Ucg.Ucv_values(coeff, cv_grid)

            label = "{}  excl $|i - j| < {}$".format(legend[i], bondcut[i])
            plt.plot(cv_grid, Ucv, label=label)

    #plt.legend(title=r"$n_{basis}$  $n_{test}$", fontsize=12)
    plt.xlabel("TIC1")
    plt.ylabel(r"$U_{cv}(\psi_1)$ (k$_B$T)")
    plt.saveplt("plots/compare_Ucv_fixed_sigma.pdf")
    plt.saveplt("plots/compare_Ucv_fixed_sigma.png")
