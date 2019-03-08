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

def scan_with_fixed_sigma(loss, Ucg, cv_r0_basis, alpha_lims, n_fixed_sigma=100):
    f_mult_12 = np.linspace(0.6, 1.35, n_fixed_sigma)**12
    alphas = np.logspace(*alpha_lims)

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

    #sigma_idx, alpha_idx = np.argwhere(vl_mse[:,:,0] == vl_mse[:,:,0].min())[0]
    #
    #new_sigma = 0.373*(f_mult_12**(1./12))

    #np.save("alpha_sigma_coeffs.npy", all_coeffs)
    #np.save("alpha_sigma_train_mse.npy", tr_mse)
    #np.save("alpha_sigma_valid_mse.npy", vl_mse)
    #np.save("scaled_sigma_vals.npy", new_sigma)
    #np.save("best_sigma_alpha_idx.npy", np.array([sigma_idx, alpha_idx]))

    #at_best_sigma = vl_mse[sigma_idx, :, 0]
    #at_best_alpha = vl_mse[:, alpha_idx, 0]

    #plt.figure()
    #plt.plot(new_sigma, at_best_alpha, 'k')
    #plt.axvline(new_sigma[sigma_idx], ls="--", color="k")
    #plt.semilogy()
    #plt.xlabel(r"Scaled radius $\sigma'$ (nm)")
    #plt.ylabel(r"Crossval score")
    #plt.savefig("cross_val_vs_sigma_fixed_alpha.pdf")
    #plt.savefig("cross_val_vs_sigma_fixed_alpha.png")

    #plt.figure()
    #for i in range(len(alphas)):
    #    plt.plot(new_sigma, vl_mse[:, i, 0], 'k')
    #plt.axvline(new_sigma[sigma_idx], ls="--", color="k")
    #plt.semilogy()
    #plt.xlabel(r"Scaled radius $\sigma'$ (nm)")
    #plt.ylabel(r"Crossval score")
    #plt.savefig("cross_val_vs_sigma_all_alpha.pdf")
    #plt.savefig("cross_val_vs_sigma_all_alpha.png")

    #plt.figure()
    #plt.plot(alphas, at_best_sigma, 'k')
    #plt.axvline(alphas[alpha_idx], ls="--", color="k")
    #plt.semilogy()
    #plt.semilogx()
    #plt.xlabel(r"Regularization $\alpha$")
    #plt.ylabel(r"Crossval score")
    #plt.savefig("cross_val_vs_alpha_fixed_sigma.pdf")
    #plt.savefig("cross_val_vs_alpha_fixed_sigma.png")

    #X, Y = np.meshgrid(new_sigma, alphas)
    #plt.figure()
    ##pcol = plt.pcolormesh(X, Y, np.log10(vl_mse[:,:,0]).T, vmin=-2.5, vmax=-2, linewidth=0, rasterized=True)
    #pcol = plt.pcolormesh(X, Y, np.log10(vl_mse[:,:,0]).T, linewidth=0, rasterized=True)
    #pcol.set_edgecolor("face")
    ##plt.semilogx()
    #plt.xlabel(r"Scaled radius $\sigma'$ (nm)")
    #plt.ylabel(r"Regulariation $\alpha$")
    #plt.semilogy()
    #cbar = plt.colorbar()
    #cbar.set_label("log(crossval score)")
    #plt.savefig("cross_val_pcolor.pdf")
    #plt.savefig("cross_val_pcolor.png")

    #min_vl_idxs = np.argwhere(vl_mse[:,:,0] <= 1.1*vl_mse[:,:,0].min())

    #if Ucg.using_cv:
    #    cv_x_vals = np.linspace(1.3*cv_r0_basis.min(), 1.2*cv_r0_basis.max(), 200).reshape((-1,1))
    #    plot_Ucv_vs_alpha_with_fixed_sigma(alpha_idx, sigma_idx, all_coeffs, alphas, new_sigma, Ucg, cv_x_vals, "fixed_sigma_")

    return f_mult_12, alphas, all_coeffs, tr_mse, vl_mse

def plot_Ucv_for_best_sigma(Ucg, cv_vals, coeff_list, alpha_list, sigma_star, prefix, ylims=None):

    coeff_star, coeff_min, coeff_max = coeff_list
    alpha_star, alpha_min, alpha_max = alpha_list

    Ucv = Ucg.Ucv_values(coeff_star, cv_vals)
    Ucv -= Ucv.min()

    Ucv_1 = Ucg.Ucv_values(coeff_min, cv_vals)
    Ucv_1 -= Ucv_1.min()

    Ucv_2 = Ucg.Ucv_values(coeff_max, cv_vals)
    Ucv_2 -= Ucv_2.min()

    #title = r"$\sigma^*={:.2f} \mathrm{{nm}}$  $\alpha^* = {:.1e}$".format(sigma_star, alpha_star)
    title = r"$\sigma^*={:.2f} \mathrm{{nm}}$".format(sigma_star)
    plt.figure()
    plt.plot(cv_vals, Ucv_1, lw=3, label=r"$\alpha_{\mathrm{min}} =" + "{:.1e}$".format(alpha_min))
    plt.plot(cv_vals, Ucv_2, lw=3, label=r"$\alpha_{\mathrm{max}} =" + "{:.1e}$".format(alpha_max))
    plt.plot(cv_vals, Ucv, 'k', lw=3, label=r"$\alpha^* =" + "{:.1e}$".format(alpha_star))

    if not ylims is None:
        plt.ylim(*ylims)

    plt.legend()
    plt.xlabel(r"TIC1 $\psi_1$")
    plt.ylabel(r"$U_{\mathrm{cv}}(\psi_1)$")
    plt.title(title)
    plt.savefig("{}compare_Ucv.pdf".format(prefix))
    plt.savefig("{}compare_Ucv.png".format(prefix))

def plot_Upair_for_best_sigma(Ucg, r_vals, coeff_list, alpha_list, sigma_star, n_pair_gauss, prefix, ylims=None, with_min=False):
    """Plot pair potential solution"""

    coeff_star, coeff_min, coeff_max = coeff_list
    alpha_star, alpha_min, alpha_max = alpha_list

    #title = r"$\sigma^*={:.2f} \mathrm{{nm}}$  $\alpha^* = {:.1e}$".format(sigma_star, alpha_max)
    title = r"$\sigma^*={:.2f} \mathrm{{nm}}$  $\alpha^* = {:.1e}$".format(sigma_star, alpha_star)
    label = r"$\alpha^*" + " = {:.1e}$".format(alpha_star)
    label_1 = r"$\alpha_{\mathrm{min}}" + " = {:.1e}$".format(alpha_min)
    label_2 = r"$\alpha_{\mathrm{max}}" + " = {:.1e}$".format(alpha_max)

    suffix = "_{}".format(Ucg.pair_symmetry)
    if with_min:
        suffix += "_with_min"

    N = Ucg.n_atoms
    bcut = Ucg.bond_cutoff
    xlims = (r_vals.min(), r_vals.max())

    xyz_traj = np.zeros((len(r_vals), 6)) 
    xyz_traj[:,4] = r_vals

    if Ucg.pair_symmetry == "shared":
        fig1 = plt.figure()
        Upair = np.zeros(len(r_vals))
        for i in range(len(coeff_star)):
            Upair += coeff_star[i]*Ucg.U_funcs[1][i](*xyz_traj.T)

        Upair_1 = np.zeros(len(r_vals))
        for i in range(len(coeff_min)):
            Upair_1 += coeff_min[i]*Ucg.U_funcs[1][i](*xyz_traj.T)

        Upair_2 = np.zeros(len(r_vals))
        for i in range(len(coeff_max)):
            Upair_2 += coeff_max[i]*Ucg.U_funcs[1][i](*xyz_traj.T)

        if with_min:
            plt.plot(r_vals, Upair_1, lw=3, label=label_1)
            plt.plot(r_vals, Upair_2, lw=3, label=label_2)

        plt.plot(r_vals, Upair,  color='k', ls="-", lw=3, label=label)

        plt.legend()
        plt.title(title, fontsize=16)

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

        fig1, axes = plt.subplots(ncols, ncols, figsize=(4*ncols, 4*ncols))
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

                    Upair_2 = np.zeros(len(r_vals))
                    Upair_2 += coeff_max[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
                    for k in range(n_pair_gauss):
                        c_k = coeff_max[c_idx_start + k + 1]
                        Upair_2 += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

                    if with_min:
                        ax.plot(r_vals, Upair_1, lw=3)
                        ax.plot(r_vals, Upair_2, lw=3)

                    ax.plot(r_vals, Upair, color='k', ls="-", lw=3)

                    ax.annotate(r"$|i - j| = {:d}$".format(seps[pot_idx]), fontsize=16,
                            xy=(0,0), xytext=(0.55, 0.7), bbox={"facecolor":"w", "alpha":1, "edgecolor":"k"},
                            xycoords="axes fraction", textcoords="axes fraction")

                if not ylims is None:
                    ax.set_ylim(*ylims)
                ax.set_xlim(*xlims)

                if j == 0:
                    ax.set_ylabel(r"$U_{\mathrm{pair}}(r_{ij})$")
                if i == (ncols - 1):
                    ax.set_xlabel(r"$r_{ij}$ (nm)")
        fig1.suptitle(title, fontsize=18)

    elif Ucg.pair_symmetry == "unique":
        coord_idxs_by_seq_sep = Ucg._generate_pairwise_idxs(bond_cutoff=Ucg.bond_cutoff, sort_by_seq_sep=True)
        n_seps = len(coord_idxs_by_seq_sep)
        ncols = int(np.ceil(np.sqrt(float(n_seps))))

        seps = []
        for i in range(N - bcut):
            for j in range(i + bcut, N):
                seps.append(j - i)

        sep_to_ax_idx = {}
        for i in range(ncols):
            for j in range(ncols):
                pot_idx = i*ncols + j
                if pot_idx < n_seps:
                    sep = seps[pot_idx]
                    if not sep in sep_to_ax_idx:
                        sep_to_ax_idx[sep] = (i,j)

        pot_idx = 0
        pair_idxs_to_c_start = {}
        for i in range(N):
            for j in range(N):
                if i < (N - bcut) and (j >= i + bcut):
                    c_idx_start = pot_idx*n_pair_gauss
                    pot_idx += 1
                    pair_idxs_to_c_start[(i,j)] = c_idx_start

        fig1, axes = plt.subplots(ncols, ncols, figsize=(4*ncols, 4*ncols))
        pot_idx = 0
        for i in range(N):
            for j in range(N):
                if i < (N - bcut) and (j >= i + bcut):
                    sep = np.abs(j - i)
                    c_idx_start = pot_idx*n_pair_gauss

                    ax_i, ax_j = sep_to_ax_idx[sep]

                    ax = axes[ax_i, ax_j]

                    # plot for this pair
                    Upair = np.zeros(len(r_vals))
                    Upair += coeff_star[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
                    for k in range(n_pair_gauss):
                        c_k = coeff_star[c_idx_start + k + 1]
                        Upair += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

                    #if with_min:
                    #    Upair_1 = np.zeros(len(r_vals))
                    #    Upair_1 += coeff_min[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
                    #    for k in range(n_pair_gauss):
                    #        c_k = coeff_min[c_idx_start + k + 1]
                    #        Upair_1 += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

                    #    Upair_2 = np.zeros(len(r_vals))
                    #    Upair_2 += coeff_max[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
                    #    for k in range(n_pair_gauss):
                    #        c_k = coeff_max[c_idx_start + k + 1]
                    #        Upair_2 += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)
                    #    ax.plot(r_vals, Upair_1, lw=3)
                    #    ax.plot(r_vals, Upair_2, lw=3)

                    #ax.plot(r_vals, Upair, 'k', lw=3)
                    ax.plot(r_vals, Upair)
                    pot_idx += 1

                    #ax.plot([-10],[-10], 'k.')

                    if not ylims is None:
                        ax.set_ylim(*ylims)
                    ax.set_xlim(*xlims)

                    if ax_j == 0:
                        ax.set_ylabel(r"$U_{\mathrm{pair}}(r_{ij})$")

                    if ax_i == (ncols - 1):
                        ax.set_xlabel(r"$r_{ij}$ (nm)")
        fig1.suptitle(title, fontsize=18)

        #for s in range(len(seps)):
        #    sep = seps[s]
        #    n_pairs_sep = N - sep
        #    ncols = int(np.ceil(np.sqrt(n_pairs_sep)))
        #    fig2, axes = plt.subplots(ncols, ncols, figsize=(4*ncols, 4*ncols)) 
        #    for i in range(n_pairs_sep):
        #        j = i + sep
        #        c_idx_start = pair_idxs_to_c_start[(i,j)]

        #        if n_pairs_sep > 1:
        #            ax_i = int(np.floor(i / float(ncols)))
        #            ax_j = int(i % ncols)
        #            ax = axes[ax_i, ax_j]
        #        else:
        #            ax = axes

        #        Upair = np.zeros(len(r_vals))
        #        Upair += coeff_star[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
        #        for k in range(n_pair_gauss):
        #            c_k = coeff_star[c_idx_start + k + 1]
        #            Upair += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

        #        ax.plot(r_vals, Upair, 'k', lw=3)

        #        ax.annotate(r"$({:d}, {:d})$".format(i + 1, j + 1), fontsize=16,
        #                xy=(0,0), xytext=(0.55, 0.7), bbox={"facecolor":"w", "alpha":1, "edgecolor":"k"},
        #                xycoords="axes fraction", textcoords="axes fraction")

        #        if not ylims is None:
        #            ax.set_ylim(*ylims)
        #        ax.set_xlim(*xlims)
        #    fig2.suptitle(r"$|i - j| ={:d}$".format(sep))
        #    fig2.savefig("{}Upair_unique_{}.pdf".format(prefix, sep))
        #    fig2.savefig("{}Upair_unique_{}.png".format(prefix, sep))
        #    plt.close(fig2)

        for s in range(len(seps)):
            sep = seps[s]
            n_pairs_sep = N - sep

            if (n_pairs_sep % 2) == 0:
                # even
                ncols = int(n_pairs_sep/2)
            else:
                # odd
                ncols = int((n_pairs_sep + 1)/2)

            fig3, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4)) 
            for i in range(ncols):
                if ncols == 1:
                    ax = axes
                else:
                    ax = axes[i]

                i1 = i
                j1 = i1 + sep
                c_idx_start = pair_idxs_to_c_start[(i1,j1)]

                Upair = np.zeros(len(r_vals))
                Upair += coeff_star[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
                for k in range(n_pair_gauss):
                    c_k = coeff_star[c_idx_start + k + 1]
                    Upair += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

                ax.plot(r_vals, Upair, lw=3, label=r"$({:d}, {:d})$".format(i1 + 1, j1 + 1))

                if (ncols % 2) == 1 and i == ncols - 1:
                    # if there is an odd number of pairs and this is the 
                    # central pair then there will be no other thing to plot.
                    pass
                else:
                    j2 = N - i - 1
                    i2 = j2 - sep - 1
                    c_idx_start = pair_idxs_to_c_start[(i2,j2)]

                    Upair = np.zeros(len(r_vals))
                    Upair += coeff_star[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
                    for k in range(n_pair_gauss):
                        c_k = coeff_star[c_idx_start + k + 1]
                        Upair += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

                    ax.plot(r_vals, Upair, lw=3, label=r"$({:d}, {:d})$".format(i2 + 1, j2 + 1))
                ax.legend(fontsize=16)

                if i == 0:
                    ax.set_ylabel(r"$U_{\mathrm{pair}}(r_{ij})$")

                ax.set_xlabel(r"$r_{ij}$ (nm)")

                if not ylims is None:
                    ax.set_ylim(*ylims)
                ax.set_xlim(*xlims)
            fig3.suptitle(r"$|i - j| ={:d}$".format(sep))
            fig3.savefig("{}Upair_unique_{}_reversed.pdf".format(prefix, sep))
            fig3.savefig("{}Upair_unique_{}_reversed.png".format(prefix, sep))
            plt.close(fig3)

    fig1.savefig("{}compare_Upair{}.pdf".format(prefix, suffix))
    fig1.savefig("{}compare_Upair{}.png".format(prefix, suffix))

