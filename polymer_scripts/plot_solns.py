from __future__ import print_function, absolute_import
import os
import argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

import implicit_force_field.polymer_scripts.util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("msm_savedir", type=str)
    parser.add_argument("--keep_dims", type=int, default=5)
    parser.add_argument("--plot_d", action="store_true")
    parser.add_argument("--plot_U", action="store_true")
    args = parser.parse_args()

    name = args.name
    msm_savedir = args.msm_savedir
    keep_dims = args.keep_dims
    plot_d = args.plot_d
    plot_U = args.plot_U

    M = 1
    n_beads = 25
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)
    n_pair_gauss = 10
    M = 1   # number of eigenvectors to use
    #fixed_bonded_terms = False
    fixed_bonded_terms = True

    using_cv = True
    using_cv_r0 = False
    using_D2 = False
    n_cross_val_sets = 5

    msm_savedir = "msm_dists"
    eg_savedir = "Ucg_eigenpair"
    fm_savedir = "Ucg_FM"
    if fixed_bonded_terms: 
        eg_savedir += "_fixed_bonds_angles"
        fm_savedir += "_fixed_bonds_angles"
    eg_savedir += "_CV_" + str(M)
    fm_savedir += "_CV_" + str(M)

    if plot_U:
        n_basis_test = [[40, 40], [40, 100], [100, 100], [100, 200]]
        #n_basis_test = [[40, 40], [40, 100], [100, 100]]
        plt.figure()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for n_basis, n_test in n_basis_test:
            print("  {} {}".format(n_basis, n_test))
            temp_eg_savedir = eg_savedir + "_{}_{}_crossval_{}".format(n_basis, n_test, n_cross_val_sets)

            eg_coeff = np.load(temp_eg_savedir + "/rdg_cstar.npy")

            D = 1./eg_coeff[-1]

            Ucg, _, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(msm_savedir, n_beads, M, beta, fixed_bonded_terms, using_cv, using_cv_r0, using_D2, n_basis, n_test)

            U = np.zeros(len(cv_r0_basis))
            for i in range(len(eg_coeff) - 1):
                U += eg_coeff[i]*Ucg.cv_U_funcs[i](cv_r0_basis[:,0])
            U -= U.min()

            #plt.plot(cv_r0_basis, U, label=r"$n_{{basis}} = {}$ $n_{{test}} = {}$".format(n_basis, n_test))
            ln1 = ax1.plot(cv_r0_basis, U, label=r"${}$   ${}$".format(n_basis, n_test))

            ax2.plot(cv_r0_basis, D*np.ones(len(cv_r0_basis)))


            temp_fm_savedir = fm_savedir + "_{}_{}_crossval_{}".format(n_basis, n_test, n_cross_val_sets)
            if os.path.exists(temp_fm_savedir + "/rdg_cstar.npy"):
                fm_coeff = np.load(temp_fm_savedir + "/rdg_cstar.npy")

                U_fm = np.zeros(len(cv_r0_basis))
                for i in range(len(fm_coeff)):
                    U_fm += fm_coeff[i]*Ucg.cv_U_funcs[i](cv_r0_basis[:,0])
                U_fm -= U_fm.min()
                ax1.plot(cv_r0_basis, U_fm, ls="--", color=ln1[0].get_color(), label=r"${}$ FM".format(n_basis))

        ax1.legend(title=r"$n_{basis}$  $n_{test}$", fontsize=12)
        ax1.set_xlabel("TIC1")
        ax1.set_ylabel(r"$U_{cg}$ (k$_B$T)")

        ax2.set_xlabel("TIC1")
        ax2.set_ylabel(r"$D$")
        fig.savefig("plots/compare_Ucg_ntest_nbasis.pdf")
        fig.savefig("plots/compare_Ucg_ntest_nbasis.png")

    if plot_d:
        n_basis_test = [[40, 40], [100, 100], [100, 200]]
        #n_basis_test = [[40, 40], [100, 100]]
        mid_bin = np.load(msm_savedir + "/psi1_mid_bin.npy")[1:-1]

        plt.figure()
        for n_basis, n_test in n_basis_test:
            print("  {} {}".format(n_basis, n_test))
            temp_eg_savedir = eg_savedir + "_{}_{}_crossval_{}".format(n_basis, n_test, n_cross_val_sets)
            coeff = np.load(temp_eg_savedir + "/rdg_cstar.npy")
            d = np.load(temp_eg_savedir + "/d.npy")
            X = np.load(temp_eg_savedir + "/X.npy")

            centers = np.linspace(mid_bin.min(), mid_bin.max(), n_test)

            plt.plot(centers, d, 'o', label=r"${}$".format(n_test))

        plt.legend(title=r"$n_{test}$", fontsize=12)
        plt.xlabel("$f_j$ center along $\psi_1$")
        plt.ylabel(r"$\langle \psi_1, \nabla U_0 \cdot \nabla f_j - kT\Delta f_j \rangle$")
        plt.savefig("plots/compare_d_ntest.pdf")
        plt.savefig("plots/compare_d_ntest.png")
