from __future__ import print_function, absolute_import
import numpy as np
import os

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.use("Agg")
import matplotlib.pyplot as plt

import implicit_force_field.polymer_scripts.util as util

def plot_Ucv_for_EG_FM_bondcuts_3_4():

    method_code = ["EG", "EG", "FM", "FM"]
    cg_method = ["eigenpair", "eigenpair", "force-matching", "force-matching"]
    bondcut = [4, 3, 4, 3]
    linesty = ["-", "--", "-", "--"]
    colors = ["k", "k", "r", "r"]
    legend = ["spectral", "spectral", "force match", "force match"] 
    a_coeff = [0.027, 0.027, None, None]

    plt.figure()
    for i in range(len(method_code)):
        print("{} |i - j| >= {}".format(legend[i], bondcut[i]))

        cg_savedir = util.Ucg_dirname(cg_method[i], M, using_U0, fix_back,
                fix_exvol, bondcut[i], using_cv, n_cv_basis_funcs=n_basis,
                n_cv_test_funcs=n_test, a_coeff=a_coeff[i])

        os.chdir(cg_savedir)
        if os.path.exists("alpha_sigma_valid_mse.npy"):
            label = r"{}  $|i - j|$ >= ${}$".format(legend[i], bondcut[i])
            vl_mse = np.load("alpha_sigma_valid_mse.npy")
            sigma_idx, alpha_idx = np.load("best_sigma_alpha_idx.npy")
            new_sigma = np.load("scaled_sigma_vals.npy")
            at_best_alpha = vl_mse[:, alpha_idx, 0]
            #plt.plot(new_sigma, at_best_alpha, color=colors[i], ls=linesty[i], label=label)
            plt.plot(new_sigma, at_best_alpha, label=label)
        os.chdir("..")

    plt.legend(loc=2)
    #plt.ylim(1e-3, 1e12)
    plt.ylim(1e-10, 1e12)
    plt.semilogy()
    plt.xlabel(r"Scaled radius $\sigma_{ex}$ (nm)")
    plt.ylabel(r"Crossval score")

    plt.savefig("plots/compare_crossval_vs_scaled_sigma.pdf")
    plt.savefig("plots/compare_crossval_vs_scaled_sigma.png")

if __name__ == "__main__":
    using_U0 = True
    using_cv = True
    using_D2 = False
    fix_back = True
    fix_exvol = False
    #a_coeff = True
    n_basis = 40
    n_test = 100
    n_pair_gauss = None
    M = 1
    n_beads = 25
    msm_savedir = "msm_dists"
    beta = 1./(0.0083145*300)

    method_code = ["EG", "EG", "FM", "FM"]
    cg_method = ["eigenpair", "eigenpair", "force-matching", "force-matching"]
    bondcut = [4, 3, 4, 3]
    linesty = ["-", "--", "-", "--"]
    colors = ["k", "k", "r", "r"]
    legend = ["spectral", "spectral", "force match", "force match"] 
    a_coeff = [0.027, 0.027, True, True]


    method_code = ["EG", "EG"]
    cg_method = ["eigenpair", "eigenpair"]
    bondcut = [4, 4]
    linesty = ["-", "--", "-", "-", "--"]
    colors = ["k", "k", "r", "r"]
    legend = ["spectral", "spectral"] 
    a_coeff = [0.027, 0.000135]

    #plt.figure()
    #for i in range(len(method_code)):
    #    print("{} |i - j| >= {}".format(legend[i], bondcut[i]))

    #    cg_savedir = util.Ucg_dirname(cg_method[i], M, using_U0, fix_back,
    #            fix_exvol, bondcut[i], using_cv, n_cv_basis_funcs=n_basis,
    #            n_cv_test_funcs=n_test, a_coeff=a_coeff[i])

    #    os.chdir(cg_savedir)
    #    if os.path.exists("alpha_sigma_valid_mse.npy"):
    #        label = r"{}  $|i - j|$ >= ${}$".format(legend[i], bondcut[i])
    #        vl_mse = np.load("alpha_sigma_valid_mse.npy")
    #        sigma_idx, alpha_idx = np.load("best_sigma_alpha_idx.npy")
    #        new_sigma = np.load("scaled_sigma_vals.npy")
    #        at_best_alpha = vl_mse[:, alpha_idx, 0]
    #        #plt.plot(new_sigma, at_best_alpha, color=colors[i], ls=linesty[i], label=label)
    #        plt.plot(new_sigma, at_best_alpha, label=label)
    #    os.chdir("..")

    #plt.legend(loc=2)
    ##plt.ylim(1e-3, 1e12)
    #plt.ylim(1e-10, 1e12)
    #plt.semilogy()
    #plt.xlabel(r"Scaled radius $\sigma_{\mathrm{ex}}$ (nm)")
    #plt.ylabel(r"Crossval score")

    #plt.savefig("plots/compare_crossval_vs_scaled_sigma_nobc_3.pdf")
    #plt.savefig("plots/compare_crossval_vs_scaled_sigma_nobc_3.png")

    plt.figure()
    for i in range(len(method_code)):
        print("{} |i - j| >= {}".format(legend[i], bondcut[i]))

        cg_savedir = util.Ucg_dirname(cg_method[i], M, using_U0, fix_back,
                fix_exvol, bondcut[i], using_cv, n_cv_basis_funcs=n_basis,
                n_cv_test_funcs=n_test, a_coeff=a_coeff[i])

        if os.path.exists(cg_savedir + "/rdg_fixed_sigma_cstar.npy"):
            coeff = np.load(cg_savedir + "/rdg_fixed_sigma_cstar.npy")

            Ucg, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(msm_savedir,
                    n_beads, M, beta, fix_back, fix_exvol, using_cv, using_D2,
                    n_basis, n_test, n_pair_gauss, bondcut[i], a_coeff=a_coeff[i])

            print("{}  {}  {}  {}".format(coeff.shape[0], Ucg.n_cart_params, Ucg.n_cv_params, Ucg.fixed_a_coeff))

            cv_grid = np.linspace(1.3*cv_r0_basis.min(), 1.2*cv_r0_basis.max(), 200)

            Ucv = Ucg.Ucv_values(coeff, cv_grid)
            Ucv -= Ucv.min()

            label = r"$a = {}$".format(a_coeff[i])
            plt.plot(cv_grid, Ucv, label=label)

    title = r"{}  $|i - j|$ >= ${}$".format(legend[i], bondcut[i])
    #plt.legend(title=r"$n_{basis}$  $n_{test}$", fontsize=12)
    plt.legend()
    plt.title(title)
    plt.xlabel("TIC1")
    plt.ylabel(r"$U_{\mathrm{cv}}(\psi_1)$")
    plt.savefig("plots/compare_Ucv_fixed_sigma_diff_a.pdf")
    plt.savefig("plots/compare_Ucv_fixed_sigma_diff_a.png")
    #plt.savefig("plots/compare_Ucv_fixed_sigma.pdf")
    #plt.savefig("plots/compare_Ucv_fixed_sigma.png")
