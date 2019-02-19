from __future__ import print_function, absolute_import
import numpy as np

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.use("Agg")
import matplotlib.pyplot as plt

def plot_train_test_mse(alphas, train_mse, test_mse, 
        xlabel=r"Regularization $\alpha$", ylabel="Mean squared error (MSE)", 
        title="", prefix=""):
    """Plot mean squared error for training and test data"""

    #sum_mse = train_mse + test_mse
    #alpha_star = alphas[np.argmin(sum_mse[:,0])]

    (test_mse < 1.10*test_mse.min())
    alpha_star = alphas[np.argmin(test_mse[:,0])]

    plt.figure()
    ln1 = plt.plot(alphas, train_mse[:,0], label="Training set")[0]
    ln2 = plt.plot(alphas, test_mse[:,0], label="Test set")[0]
    plt.fill_between(alphas, train_mse[:,0] + train_mse[:,1],
            y2=train_mse[:,0] - train_mse[:,1],
            facecolor=ln1.get_color(), alpha=0.5)

    plt.fill_between(alphas, test_mse[:,0] + test_mse[:,1],
            y2=test_mse[:,0] - test_mse[:,1],
            facecolor=ln2.get_color(), alpha=0.5)

    plt.axvline(alpha_star, color='k', ls='--', label=r"$\alpha^* = {:.2e}$".format(alpha_star))
    plt.semilogx(True)
    plt.semilogy(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(prefix + "train_test_mse.pdf")
    plt.savefig(prefix + "train_test_mse.png")


def plot_Ucg_vs_alpha_with_fixed_sigma(alpha_idx, sigma_idx, coeffs, alphas, new_sigma, Ucg, cv_r0, prefix, ylim=None):

    plt.figure()
    coeff = coeffs[sigma_idx, alpha_idx]
    U = np.zeros(len(cv_r0))
    for i in range(Ucg.n_cv_params):
        U += coeff[i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
    U -= U.min()
    plt.plot(cv_r0[:,0], U, color='k', lw=3)

    if not ylim is None:
        plt.ylim(0, ylim)

    plt.title(r"$\alpha^*={:.0e}$  $\sigma^* = {:.3f}$ (nm)".format(alphas[alpha_idx], new_sigma[sigma_idx]))
    #plt.title(r"$\sigma^* = {:.3f}$ (nm)".format(alphas[alpha_idx], new_sigma[sigma_idx]))
    plt.xlabel(r"TIC1 $\psi_1$")
    plt.ylabel(r"$U_{cg}(\psi_1)$ (kJ/mol)")
    plt.savefig("{}Ucv.pdf".format(prefix))
    plt.savefig("{}Ucv.png".format(prefix))

def scan_with_fixed_sigma(loss, Ucg, cv_r0_basis):
    f_mult_12 = np.linspace(0.6, 1.35, 100)**12
    alphas = np.logspace(-10, 2,50)

    fix_ck_idxs = np.array([0])
    all_coeffs = []
    all_train_mse = []
    all_valid_mse = []
    for i in range(len(f_mult_12)):
        fix_ck_vals = np.array([f_mult_12[i]])
        coeffs, train_mse, valid_mse = loss.solve_with_fixed_params(alphas, fix_ck_idxs, fix_ck_vals)
        all_coeffs.append(coeffs)
        all_train_mse.append(train_mse)
        all_valid_mse.append(valid_mse)

    all_coeffs = np.array(all_coeffs)
    tr_mse = np.array(all_train_mse)
    vl_mse = np.array(all_valid_mse)

    new_sigma = 0.373*(f_mult_12**(1./12))
    plt.figure()
    for i in range(len(alphas)):
        plt.plot(new_sigma, vl_mse[:, i, 0])
    plt.semilogy()
    plt.xlabel(r"Scaled radius $\sigma'$ (nm)")
    plt.ylabel(r"Crossval score")
    plt.savefig("cross_val_vs_sigma_fixed_alpha.pdf")
    plt.savefig("cross_val_vs_sigma_fixed_alpha.png")

    plt.figure()
    for i in range(len(f_mult_12)):
        plt.plot(alphas, vl_mse[i, :, 0])
    plt.semilogy()
    plt.semilogx()
    plt.savefig("cross_val_vs_alpha_fixed_sigma.pdf")
    plt.savefig("cross_val_vs_alpha_fixed_sigma.png")

    X, Y = np.meshgrid(new_sigma, alphas)
    plt.figure()
    pcol = plt.pcolormesh(X, Y, np.log10(vl_mse[:,:,0]).T, vmin=-2.5, vmax=-2, linewidth=0, rasterized=True)
    pcol.set_edgecolor("face")
    #plt.semilogx()
    plt.xlabel(r"Scaled radius $\sigma'$ (nm)")
    plt.ylabel(r"Regulariation $\alpha$")
    plt.semilogy()
    cbar = plt.colorbar()
    cbar.set_label("log(crossval score)")
    plt.savefig("cross_val_pcolor.pdf")
    plt.savefig("cross_val_pcolor.png")

    sigma_idx, alpha_idx = np.argwhere(vl_mse[:,:,0] == vl_mse[:,:,0].min())[0]
    cv_x_vals = np.linspace(1.3*cv_r0_basis.min(), 1.2*cv_r0_basis.max(), 200).reshape((-1,1))

    plot_Ucg_vs_alpha_with_fixed_sigma(alpha_idx, sigma_idx, all_coeffs, alphas, new_sigma, Ucg, cv_x_vals, "fixed_sigma_")

    return f_mult_12, alphas, all_coeffs, tr_mse, vl_mse
