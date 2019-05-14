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

def compare_Ucv_for_free_a_coeff():
    n_basis_test = [[40, 40], [40, 100], [100, 100], [100, 200]]
    #n_basis_test = [[40, 40], [40, 100], [100, 100]]
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for n_basis, n_test in n_basis_test:
        print("  {} {}".format(n_basis, n_test))
        temp_eg_savedir = eg_savedir + "_{}_{}_crossval_{}".format(n_basis, n_test, n_cross_val_sets)

        eg_coeff = np.load(temp_eg_savedir + "/rdg_cstar.npy")

        D = 1./eg_coeff[-1]

        cg_savedir = util.Ucg_dirname(cg_method, M, using_U0, fix_back, fix_exvol,
                bond_cutoff, using_cv, n_cv_basis_funcs=n_basis,
                n_cv_test_funcs=n_test, a_coeff=a_coeff)

        Ucg, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(
                msm_savedir, n_beads, M, beta, fix_back, fix_exvol, using_cv,
                using_D2, n_basis, n_test, n_pair_gauss,
                bond_cutoff)

        U = Ucg.Ucv_values(coeff, cv_r0_basis[:,0])

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

def plot_Ucv_crossval_sigma():
    # plot collective variable potentials
    n_pair_gauss = None
    lin_pot = True
    fixed_a = True
    using_U0 = True
    using_cv = True
    fix_back = True
    fix_exvol = False
    using_cv_r0 = False
    using_D2 = False
    n_cross_val_sets = 5
    pair_symmetry = None
    n_cv_test_funcs = 100
    n_cv_basis_funcs = 40
    bond_cutoff = 4

    methods = ["force-matching", "eigenpair", "eigenpair"]
    a_coeffs = [1, 0.027, 0.000135]

    plt.figure()
    for i in range(len(methods)):
        cg_method = methods[i]
        a_coeff = a_coeffs[i]

        cg_savedir = util.Ucg_dirname(cg_method, M, using_U0, fix_back,
                fix_exvol, bond_cutoff, using_cv,
                n_cv_basis_funcs=n_cv_basis_funcs, n_cv_test_funcs=n_cv_test_funcs,
                a_coeff=a_coeff, n_pair_gauss=n_pair_gauss, cv_lin_pot=lin_pot,
                pair_symmetry=pair_symmetry)

        print(cg_savedir)

        Ucg, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg( msm_savedir,
                n_beads, M, beta, fix_back, fix_exvol, using_cv, using_D2,
                n_cv_basis_funcs, n_cv_test_funcs, n_pair_gauss, bond_cutoff,
                cv_lin_pot=lin_pot, a_coeff=a_coeff, pair_symmetry=pair_symmetry)

        cv_grid = np.linspace(1.3*cv_r0_basis.min(), 1.2*cv_r0_basis.max(), 200)

        coeff = np.load(cg_savedir + "/rdg_fixed_sigma_cstar.npy")
        Ucv = Ucg.Ucv_values(coeff, cv_grid)
        Ucv -= Ucv.min()

        if cg_method == "force-matching":
            label = "FM"
        else:
            label = r"EG $a = {:.2e}$".format(a_coeff)

        plt.plot(cv_grid, beta*Ucv, label=label, lw=2)

    #plt.legend(title=r"$n_{basis}$  $n_{test}$", fontsize=12)
    plt.legend()
    plt.xlabel("TIC1 $\psi_1$")
    plt.ylabel(r"$U_{\mathrm{cv}}(\psi_1)$ (k$_B$T)")
    plt.savefig("plots/Ucv_crossval_sigma.pdf")
    plt.savefig("plots/Ucv_crossval_sigma.png")

def plot_Upair_crossval_sigma():
    # plot collective variable potentials
    n_pair_gauss = 40
    lin_pot = False
    fixed_a = True
    using_U0 = True
    using_cv = False
    fix_back = True
    fix_exvol = False
    using_cv_r0 = False
    using_D2 = False
    n_cross_val_sets = 5
    pair_symmetry = "shared"
    n_cv_test_funcs = 100
    n_cv_basis_funcs = 40
    bond_cutoff = 4

    r_vals = np.linspace(0.1, 2, 300)

    methods = ["force-matching", "eigenpair", "eigenpair"]
    a_coeffs = [1, 0.027, 0.000135]
    plt.figure()
    for i in range(len(methods)):
        cg_method = methods[i]
        a_coeff = a_coeffs[i]

        cg_savedir = util.Ucg_dirname(cg_method, M, using_U0, fix_back,
                fix_exvol, bond_cutoff, using_cv,
                n_cv_basis_funcs=n_cv_basis_funcs, n_cv_test_funcs=n_cv_test_funcs,
                a_coeff=a_coeff, n_pair_gauss=n_pair_gauss, cv_lin_pot=lin_pot,
                pair_symmetry=pair_symmetry)

        print(cg_savedir)

        Ucg, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg( msm_savedir,
                n_beads, M, beta, fix_back, fix_exvol, using_cv, using_D2,
                n_cv_basis_funcs, n_cv_test_funcs, n_pair_gauss, bond_cutoff,
                cv_lin_pot=lin_pot, a_coeff=a_coeff, pair_symmetry=pair_symmetry)


        coeff = np.load(cg_savedir + "/rdg_fixed_sigma_cstar.npy")
        Upair = Ucg.Upair_values(coeff, r_vals)[0]
        #Upair -= Upair.min()

        if cg_method == "force-matching":
            label = "FM"
        else:
            label = r"EG $a = {:.2e}$".format(a_coeff)

        plt.plot(r_vals, beta*Upair, label=label, lw=2)

    plt.xlim(0.25, 1.25)
    plt.ylim(-1, 3)
    plt.legend()
    plt.xlabel("$r$ (nm)")
    plt.ylabel(r"$U_{\mathrm{pair}}(r)$ (k$_B$T)")
    plt.savefig("plots/Upair_crossval_sigma.pdf")
    plt.savefig("plots/Upair_crossval_sigma.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("msm_savedir", type=str)
    #parser.add_argument("--keep_dims", type=int, default=5)
    parser.add_argument("--plot_d", action="store_true")
    parser.add_argument("--plot_U", action="store_true")
    args = parser.parse_args()

    name = args.name
    msm_savedir = args.msm_savedir
    #keep_dims = args.keep_dims
    plot_d = args.plot_d
    plot_U = args.plot_U

    M = 1
    n_beads = 25
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)
    M = 1   

    msm_savedir = "msm_dists"

    #plot_Upair_crossval_sigma()
    plot_Ucv_crossval_sigma()

    #if plot_U:
    #    n_test = 100
    #    n_basis = 40

    #    plt.figure()
    #    for i in range(len(method)):

    #        cg_savedir = util.Ucg_dirname(cg_method, M, using_U0, fix_back, fix_exvol,
    #                bond_cutoff, using_cv, n_cv_basis_funcs=n_basis,
    #                n_cv_test_funcs=n_test, a_coeff=a_coeff)

    #        Ucg, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(
    #                msm_savedir, n_beads, M, beta, fix_back, fix_exvol, using_cv,
    #                using_D2, n_basis, n_test, n_pair_gauss,
    #                bond_cutoff)

    #        cv_grid = np.linspace(1.3*cv_r0_basis.min(), 1.2*cv_r0_basis.max(), 200).reshape((-1,1))

    #        coeff = np.load(cg_savedir + "/rdg_fixed_sigma_cstar.npy")
    #        Ucv = Ucg.Ucv_values(coeff, cv_grid)

    #        plt.plot(cv_grid, Ucv, label=r"${}$".format(method[i]))

    #    plt.legend(title=r"$n_{basis}$  $n_{test}$", fontsize=12)
    #    plt.xlabel("TIC1")
    #    plt.ylabel(r"$U_{cv}(\psi_1)$ (k$_B$T)")
    #    plt.saveplt("plots/compare_Ucv_fixed_sigma.pdf")
    #    plt.saveplt("plots/compare_Ucv_fixed_sigma.png")

    #if plot_d:
    #    n_basis_test = [[40, 40], [100, 100], [100, 200]]
    #    #n_basis_test = [[40, 40], [100, 100]]
    #    mid_bin = np.load(msm_savedir + "/psi1_mid_bin.npy")[1:-1]

    #    plt.figure()
    #    for n_basis, n_test in n_basis_test:
    #        print("  {} {}".format(n_basis, n_test))
    #        temp_eg_savedir = eg_savedir + "_{}_{}_crossval_{}".format(n_basis, n_test, n_cross_val_sets)
    #        coeff = np.load(temp_eg_savedir + "/rdg_cstar.npy")
    #        d = np.load(temp_eg_savedir + "/d.npy")
    #        X = np.load(temp_eg_savedir + "/X.npy")

    #        centers = np.linspace(mid_bin.min(), mid_bin.max(), n_test)

    #        plt.plot(centers, d, 'o', label=r"${}$".format(n_test))

    #    plt.legend(title=r"$n_{test}$", fontsize=12)
    #    plt.xlabel("$f_j$ center along $\psi_1$")
    #    plt.ylabel(r"$\langle \psi_1, \nabla U_0 \cdot \nabla f_j - kT\Delta f_j \rangle$")
    #    plt.savefig("plots/compare_d_ntest.pdf")
    #    plt.savefig("plots/compare_d_ntest.png")
