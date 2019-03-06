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


def plot_Ucv_vs_alpha_with_fixed_sigma(alpha_idx, sigma_idx, coeffs, alphas, new_sigma, Ucg, cv_r0, prefix, ylim=None):

    plt.figure()
    coeff = coeffs[sigma_idx, alpha_idx]
    #Ucg.Ucv_values()
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

    sigma_idx, alpha_idx = np.argwhere(vl_mse[:,:,0] == vl_mse[:,:,0].min())[0]
    
    new_sigma = 0.373*(f_mult_12**(1./12))

    np.save("alpha_sigma_coeffs.npy", all_coeffs)
    np.save("alpha_sigma_train_mse.npy", tr_mse)
    np.save("alpha_sigma_valid_mse.npy", vl_mse)
    np.save("scaled_sigma_vals.npy", new_sigma)
    np.save("best_sigma_alpha_idx.npy", np.array([sigma_idx, alpha_idx]))

    at_best_sigma = vl_mse[sigma_idx, :, 0]
    at_best_alpha = vl_mse[:, alpha_idx, 0]

    plt.figure()
    plt.plot(new_sigma, at_best_alpha, 'k')
    plt.axvline(new_sigma[sigma_idx], ls="--", color="k")
    plt.semilogy()
    plt.xlabel(r"Scaled radius $\sigma'$ (nm)")
    plt.ylabel(r"Crossval score")
    plt.savefig("cross_val_vs_sigma_fixed_alpha.pdf")
    plt.savefig("cross_val_vs_sigma_fixed_alpha.png")

    plt.figure()
    for i in range(len(alphas)):
        plt.plot(new_sigma, vl_mse[:, i, 0], 'k')
    plt.axvline(new_sigma[sigma_idx], ls="--", color="k")
    plt.semilogy()
    plt.xlabel(r"Scaled radius $\sigma'$ (nm)")
    plt.ylabel(r"Crossval score")
    plt.savefig("cross_val_vs_sigma_all_alpha.pdf")
    plt.savefig("cross_val_vs_sigma_all_alpha.png")

    plt.figure()
    plt.plot(alphas, at_best_sigma, 'k')
    plt.axvline(alphas[alpha_idx], ls="--", color="k")
    plt.semilogy()
    plt.semilogx()
    plt.xlabel(r"Regularization $\alpha$")
    plt.ylabel(r"Crossval score")
    plt.savefig("cross_val_vs_alpha_fixed_sigma.pdf")
    plt.savefig("cross_val_vs_alpha_fixed_sigma.png")

    X, Y = np.meshgrid(new_sigma, alphas)
    plt.figure()
    #pcol = plt.pcolormesh(X, Y, np.log10(vl_mse[:,:,0]).T, vmin=-2.5, vmax=-2, linewidth=0, rasterized=True)
    pcol = plt.pcolormesh(X, Y, np.log10(vl_mse[:,:,0]).T, linewidth=0, rasterized=True)
    pcol.set_edgecolor("face")
    #plt.semilogx()
    plt.xlabel(r"Scaled radius $\sigma'$ (nm)")
    plt.ylabel(r"Regulariation $\alpha$")
    plt.semilogy()
    cbar = plt.colorbar()
    cbar.set_label("log(crossval score)")
    plt.savefig("cross_val_pcolor.pdf")
    plt.savefig("cross_val_pcolor.png")

    #min_vl_idxs = np.argwhere(vl_mse[:,:,0] <= 1.1*vl_mse[:,:,0].min())

    if Ucg.using_cv:
        cv_x_vals = np.linspace(1.3*cv_r0_basis.min(), 1.2*cv_r0_basis.max(), 200).reshape((-1,1))
        plot_Ucv_vs_alpha_with_fixed_sigma(alpha_idx, sigma_idx, all_coeffs, alphas, new_sigma, Ucg, cv_x_vals, "fixed_sigma_")

    return f_mult_12, alphas, all_coeffs, tr_mse, vl_mse

def plot_Ucv_for_best_sigma(Ucg, cv_vals, coeff_star, coeff_min, sigma_star, alpha_max, alpha_min, prefix, ylims=None):

    Ucv = Ucg.Ucv_values(coeff_star, cv_vals)
    Ucv -= Ucv.min()

    Ucv_1 = Ucg.Ucv_values(coeff_min, cv_vals)
    Ucv_1 -= Ucv_1.min()

    #title = r"$\sigma^*={:.2f} \mathrm{{nm}}$  $\alpha^* = {:.1e}$".format(sigma_star, alpha_star)
    title = r"$\sigma^*={:.2f} \mathrm{{nm}}$".format(sigma_star)
    plt.figure()
    plt.plot(cv_vals, Ucv, 'k', lw=3, label=r"$\alpha_{\mathrm{max}} =" + "{:.1e}$".format(alpha_max))
    plt.plot(cv_vals, Ucv_1, 'k--', lw=3, label=r"$\alpha_{\mathrm{min}} =" + "{:.1e}$".format(alpha_min))

    if not ylims is None:
        plt.ylim(*ylims)

    plt.legend()
    plt.xlabel(r"TIC1 $\psi_1$")
    plt.ylabel(r"$U_{\mathrm{cv}}(\psi_1)$")
    plt.title(title)
    plt.savefig("{}compare_Ucv.pdf".format(prefix))
    plt.savefig("{}compare_Ucv.png".format(prefix))

def plot_Upair_for_best_sigma(Ucg, r_vals, coeff_star, coeff_min, sigma_star, alpha_max, alpha_min, n_pair_gauss, prefix, ylims=None):
    """Plot pair potential solution"""

    #title = r"$\sigma^*={:.2f} \mathrm{{nm}}$  $\alpha^* = {:.1e}$".format(sigma_star, alpha_max)
    title = r"$\sigma^*={:.2f} \mathrm{{nm}}$  $\alpha^* = {:.1e}$".format(sigma_star, alpha_max)
    label = r"$\alpha_{\mathrm{max}}" + " = {:.1e}$".format(alpha_max)
    label_1 = r"$\alpha_{\mathrm{min}}" + " = {:.1e}$".format(alpha_min)

    N = Ucg.n_atoms
    bcut = Ucg.bond_cutoff
    xlims = (r_vals.min(), r_vals.max())

    xyz_traj = np.zeros((len(r_vals), 6)) 
    xyz_traj[:,4] = r_vals

    if Ucg.pair_symmetry == "shared":
        plt.figure()
        Upair = np.zeros(len(r_vals))
        for i in range(len(coeff_star)):
            Upair += coeff_star[i]*Ucg.U_funcs[1][i](*xyz_traj.T)

        Upair_1 = np.zeros(len(r_vals))
        for i in range(len(coeff_min)):
            Upair_1 += coeff_min[i]*Ucg.U_funcs[1][i](*xyz_traj.T)

        plt.plot(r_vals, Upair,  color='k', ls="-", lw=3, label=label)
        plt.plot(r_vals, Upair_1, color='k', ls='--', lw=3, label=label_1)
        plt.title(title, fontsize=16)
        plt.legend()

        if not ylims is None:
            plt.ylim(*ylims)

        plt.xlim(*xlims)
        plt.xlabel(r"$r_{ij}$ (nm)")
        plt.ylabel(r"$U_{\mathrm{pair}}(r_{ij})$")

    elif Ucg.pair_symmetry == "seq_sep":
        coord_idxs_by_seq_sep = Ucg._generate_pairwise_idxs(bond_cutoff=Ucg.bond_cutoff, sort_by_seq_sep=True)
        n_pots = len(coord_idxs_by_seq_sep)
        ncols = int(np.ceil(np.sqrt(float(n_pots))))

        seps = [ ]
        for i in range(N - bcut):
            for j in range(i + bcut, N):
                seps.append(j - i)

        fig, axes = plt.subplots(ncols, ncols, figsize=(4*ncols, 4*ncols))
        for i in range(ncols):
            for j in range(ncols):
                ax = axes[i,j] 
                pot_idx = i*ncols + j
                if pot_idx >= n_pots:
                    ax.plot([-10],[-10], 'k.')
                else:
                    sep = seps[pot_idx]
                    c_idx_start = pot_idx*n_pair_gauss

                    Upair = np.zeros(len(r_vals))
                    Upair += coeff_star[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
                    for k in range(n_pair_gauss):
                        c_k = coeff_star[c_idx_start + k + 1]
                        Upair += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

                    Upair_1 = np.zeros(len(r_vals))
                    Upair_1 += coeff_min[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
                    for k in range(n_pair_gauss):
                        c_k = coeff_min[c_idx_start + k + 1]
                        Upair_1 += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

                    ax.plot(r_vals, Upair, color='k', ls="-", lw=3)
                    ax.plot(r_vals, Upair_1, color='k', ls='--', lw=3)

                    ax.annotate(r"$|i - j| = {:d}$".format(seps[pot_idx]), fontsize=16,
                            xy=(0,0), xytext=(0.55, 0.7), 
                            xycoords="axes fraction", textcoords="axes fraction")

                if not ylims is None:
                    ax.set_ylim(*ylims)
                ax.set_xlim(*xlims)

                if j == 0:
                    ax.set_ylabel(r"$U_{\mathrm{pair}}(r_{ij})$")
                if i == (ncols - 1):
                    ax.set_xlabel(r"$r_{ij}$ (nm)")
        fig.suptitle(title, fontsize=18)

    elif Ucg.pair_symmetry == "unique":
        coord_idxs_by_seq_sep = Ucg._generate_pairwise_idxs(bond_cutoff=Ucg.bond_cutoff)

        fig, axes = plt.subplots(N, N, figsize=(4*N, 4*N), sharex=True)
        pot_idx = 0
        for i in range(N):
            for j in range(N):
                ax = axes[i,j]
                if i < (N - bcut) and (j >= i + bcut):
                    sep = np.abs(j - i)
                    c_idx_start = pot_idx*n_pair_gauss

                    # plot for this pair
                    Upair = np.zeros(len(r_vals))
                    Upair += coeff_star[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
                    for i in range(n_pair_gauss):
                        c_k = coeff_star[c_idx_start + k + 1]
                        Upair += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

                    ax.plot(r_vals, Upair, 'k', lw=3)
                    pot_idx += 1
                else:
                    ax.plot([-10],[-10], 'k.')

                if not ylims is None:
                    ax.set_ylim(*ylims)
                ax.set_xlim(*xlims)

                if j == 0:
                    ax.set_ylabel(r"$U_{\mathrm{pair}}(r_{ij})$")

                if i == (N - 1):
                    ax.set_xlabel(r"$r_{ij}$ (nm)")
        fig.suptitle(title, fontsize=18)

    plt.savefig("{}compare_Upair_{}.pdf".format(prefix, Ucg.pair_symmetry))
    plt.savefig("{}compare_Upair_{}.png".format(prefix, Ucg.pair_symmetry))
