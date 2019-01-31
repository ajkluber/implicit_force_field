from __future__ import print_function
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import implicit_force_field.polymer_scripts.util as util

if __name__ == "__main__":
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
    #n_cv_basis_funcs = 40
    #n_cv_test_funcs = 100
    #n_cv_basis_funcs = 100
    #n_cv_test_funcs = 200

    using_D2 = False
    n_cross_val_sets = 5

    msm_savedir = "msm_dists"
    cg_savedir = "Ucg_eigenpair"
    if fixed_bonded_terms: 
        cg_savedir += "_fixed_bonds_angles"
    cg_savedir += "_CV_" + str(M)

    n_basis_test = [[40, 100], [100, 100], [100, 200]]
    plt.figure()
    for n_basis, n_test in n_basis_test:
        print("  {} {}".format(n_basis, n_test))
        savedir = cg_savedir + "_{}_{}_crossval_{}".format(n_basis, n_test, n_cross_val_sets)
        coeff = np.load(savedir + "/rdg_cstar.npy")

        Ucg, _, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(msm_savedir, n_beads, M, beta, fixed_bonded_terms, using_cv, using_cv_r0, using_D2, n_basis, n_test)

        U = np.zeros(len(cv_r0_basis))
        for i in range(len(coeff) - 1):
            U += coeff[i]*Ucg.cv_U_funcs[i](cv_r0_basis[:,0])
        U -= U.min()

        plt.plot(cv_r0_basis, U, label=r"$n_{{basis}} = {}$ $n_{{test}} = {}$".format(n_basis, n_test))

    plt.legend()
    plt.xlabel("TIC1")
    plt.ylabel("CG Potential")
    plt.savefig("plots/compare_Ucg_ntest_nbasis.pdf")
    plt.savefig("plots/compare_Ucg_ntest_nbasis.png")
