from __future__ import print_function, absolute_import
import os
import glob
import sys
import argparse
import numpy as np
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.use("Agg")
import matplotlib.pyplot as plt

import mdtraj as md

import simulation.openmm as sop

import implicit_force_field as iff
import implicit_force_field.polymer_scripts.util as util
import implicit_force_field.loss_functions as loss


def plot_Ucg_vs_psi1(coeff, Ucg, cv_r0, prefix):

    U = np.zeros(len(cv_r0))
    for i in range(len(coeff) - 1):
        U += coeff[i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
    U -= U.min()

    plt.figure()
    plt.plot(cv_r0[:,0], U)
    plt.xlabel(r"TIC1 $\psi_1$")
    plt.ylabel(r"$U_{cg}(\psi_1)$")
    plt.savefig("{}U_cv.pdf".format(prefix))
    plt.savefig("{}U_cv.png".format(prefix))


def plot_Ucg_vs_alpha(idxs, idx_star, coeffs, alphas, Ucg, cv_r0, prefix, ylim=None, fixed_a=False):

    if fixed_a:
        plt.figure()
        for n in range(len(idxs)):
            coeff = coeffs[idxs[n]]
            U = np.zeros(len(cv_r0))
            for i in range(Ucg.n_cv_params):
                U += coeff[Ucg.n_cart_params + i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
            U -= U.min()

            plt.plot(cv_r0[:,0], U, label=r"$\alpha={:.2e}$".format(alphas[idxs[n]]))

        coeff = coeffs[idx_star]
        U = np.zeros(len(cv_r0))
        for i in range(Ucg.n_cv_params):
            U += coeff[Ucg.n_cart_params + i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
        U -= U.min()
        plt.plot(cv_r0[:,0], U, color='k', lw=3, label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))

        if not ylim is None:
            plt.ylim(0, ylim)

        plt.legend()
        plt.xlabel(r"TIC1 $\psi_1$")
        plt.ylabel(r"$U_{cg}(\psi_1)$")
        plt.savefig("{}compare_Ucv.pdf".format(prefix))
        plt.savefig("{}compare_Ucv.png".format(prefix))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for n in range(len(idxs)):

            coeff = coeffs[idxs[n]]

            U = np.zeros(len(cv_r0))
            for i in range(len(coeff) - 1):
                U += coeff[i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
            U -= U.min()

            D = 1./coeff[-1]
            ax1.plot(cv_r0[:,0], U, label=r"$\alpha={:.2e}$".format(alphas[idxs[n]]))
            ax2.plot(cv_r0[:,0], D*np.ones(len(cv_r0[:,0])))

        coeff = coeffs[idx_star]
        U = np.zeros(len(cv_r0))
        for i in range(len(coeff) - 1):
            U += coeff[i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
        U -= U.min()
        D = 1./coeff[-1]
        ax1.plot(cv_r0[:,0], U, color='k', lw=3, label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))
        ax2.plot(cv_r0[:,0], D*np.ones(len(cv_r0[:,0])), color='k', lw=3)
        ax2.semilogy(True)

        if not ylim is None:
            #ax1.set_ylim(0, 500)
            ax1.set_ylim(0, ylim)

        ax1.legend()
        ax1.set_xlabel(r"TIC1 $\psi_1$")
        ax1.set_ylabel(r"$U_{cg}(\psi_1)$")
        ax2.set_xlabel(r"TIC1 $\psi_1$")
        ax2.set_ylabel(r"$D$")
        fig.savefig("{}compare_Ucv.pdf".format(prefix))
        fig.savefig("{}compare_Ucv.png".format(prefix))

def plot_Upair_vs_alpha(idxs, idx_star, coeffs, alphas, Ucg, r_vals, n_pair_gauss, prefix, ylims=None, fixed_a=False):
    """Plot pair potential solution"""

    N = Ucg.n_atoms
    bcut = Ucg.bond_cutoff
    xlims = (r_vals.min(), r_vals.max())

    xyz_traj = np.zeros((len(r_vals), 6)) 
    xyz_traj[:,4] = r_vals

    if Ucg.pair_symmetry == "shared":
        plt.figure()
        for n in range(len(idxs)):
            coeff = coeffs[idxs[n]]
            Upair = np.zeros(len(r_vals))
            for i in range(len(coeff)):
                Upair += coeff[i]*Ucg.U_funcs[1][i](*xyz_traj.T)

            plt.plot(r_vals, Upair, label=r"$\alpha={:.2e}$".format(alphas[idxs[n]]))

        coeff = coeffs[idx_star]
        Upair = np.zeros(len(r_vals))
        for i in range(len(coeff)):
            Upair += coeff[i]*Ucg.U_funcs[1][i](*xyz_traj.T)
        plt.plot(r_vals, Upair, color='k', lw=3, label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))

        if not ylims is None:
            plt.ylim(*ylims)

        plt.xlims(*xlims)

        plt.xlabel(r"$r_{ij}$ (nm)")
        plt.ylabel(r"$U_{\mathrm{pair}}(r_{ij})$")
    elif Ucg.pair_symmetry == "seq_sep":
        coord_idxs_by_seq_sep = Ucg._generate_pairwise_idxs(bond_cutoff=Ucg.bond_cutoff, sort_by_seq_sep=True)
        n_pots = len(coord_idxs_by_seq_sep)

        seps = [ ]
        for i in range(N - bcut):
            for j in range(i + bcut, N):
                seps.append(j - i)

        fig, axes = plt.subplots(1, n_pots, figsize=(n_pots*4, 4))
        for i in range(n_pots):
            ax = axes[i]
            sep = seps[i]

            c_idx_start = i*n_pair_gauss

            # plot for each regularization
            for n in range(len(idxs)):
                coeff = coeffs[idxs[n]]

                Upair = np.zeros(len(r_vals))
                Upair += coeff[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
                for k in range(n_pair_gauss):
                    c_k = coeff[c_idx_start + k + 1]
                    Upair += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

                if i == 0:
                    ax.plot(r_vals, Upair, label=r"$\alpha^*={:.2e}$".format(alphas[idxs[n]]))
                else:
                    ax.plot(r_vals, Upair)

            coeff = coeffs[idx_star]
            Upair += coeff[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
            for k in range(n_pair_gauss):
                c_k = coeff[c_idx_start + k + 1]
                Upair += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

            if i == 0:
                ax.plot(r_vals, Upair, color='k', lw=3, label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))
                ax.legend()
            else:
                ax.plot(r_vals, Upair, color='k', lw=3)

            if not ylims is None:
                ax.set_ylim(*ylims)
            ax.set_xlim(*xlims)
            

    elif Ucg.pair_symmetry == "unique":
        raise NotImplementedError

        coord_idxs_by_seq_sep = Ucg._generate_pairwise_idxs(bond_cutoff=Ucg.bond_cutoff)

        fig, axes = plt.subplots(N, N, figsize=(4*N, 4*N), sharex=True)
        for i in range(N):
            for j in range(N):
                if j >= i + bcut:
                    # for seq sep |j - i|
                    ax = axes[i,j]

                    sep = np.abs(j - i)

                    # plot for each regularization
                    for n in range(len(idxs)):
                        coeff = coeffs[idxs[n]]
                        c_idx = 0

                        coeff[sep]
                        for i in range(len(coeff) - 1):
                            ax = axes[i, j]

                            Upair = np.zeros(len(r_vals))
                            # excluded volume
                            Upair += coeff[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
                            for i in range(len(coeff)):
                                Upair += coeff[i]*Ucg.U_funcs[1][i](*xyz_traj.T)

                            #ax.plot(r_vals, Upair, label=r"$\alpha={:.2e}$".format(alphas[idxs[n]]))
                            ax.plot(r_vals, Upair)

                    coeff = coeffs[idx_star]
                    Upair = np.zeros(len(r_vals))
                    for i in range(len(coeff)):
                        Upair += coeff[i]*Ucg.U_funcs[1][i](*xyz_traj.T)
                    plt.plot(r_vals, Upair, color='k', lw=3, label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))

                    if not ylims is None:
                        ax.set_ylim(*ylims)
                    ax.set_xlim(*xlims)
                     
                else:
                    # remove axis junk to declutter
                    pass

    plt.savefig("{}compare_Upair.pdf".format(prefix))
    plt.savefig("{}compare_Upair.png".format(prefix))

def plot_Upair_for_best_sigma(Ucg, r_vals, coeffs, sigma_star, alpha_star, n_pair_gauss, prefix, ylims=None):
    """Plot pair potential solution"""

    #alpha_reg_idx = -10
    #sigma_idx, _ = np.argwhere(vl_mse[:,:,0] == vl_mse[:,:,0].min())[0]
    #sig_star = scaled_sigma[sigma_idx]
    #if sig_star < 0.3:
    #    plot_sigs = [sig_star, 0.3, 0.45]
    #    star_idx = 0
    #elif 0.3 <= sig_star <= 0.45:
    #    plot_sigs = [0.3, sig_star, 0.45]
    #    star_idx = 1
    #else:
    #    plot_sigs = [0.3, 0.45, sig_star]
    #    star_idx = 2

    #plot_idxs = [ np.argwhere((scaled_sigma - x) >= 0)[:,0][0] for x in plot_sigs ]
    #plot_idxs = [new_sigma_idx]
    #plot_sigs = [scaled_sigma[new_sigma_idx]]

    N = Ucg.n_atoms
    bcut = Ucg.bond_cutoff
    xlims = (r_vals.min(), r_vals.max())

    xyz_traj = np.zeros((len(r_vals), 6)) 
    xyz_traj[:,4] = r_vals

    if Ucg.pair_symmetry == "shared":
        coeff = coeffs[plot_idxs[n], new_alpha_idx]

        plt.figure()
        for n in range(len(idxs)):
            coeff = coeffs[idxs[n]]
            Upair = np.zeros(len(r_vals))
            for i in range(len(coeff)):
                Upair += coeff[i]*Ucg.U_funcs[1][i](*xyz_traj.T)

            plt.plot(r_vals, Upair, label=r"$\alpha={:.2e}$".format(alphas[idxs[n]]))

        coeff = coeffs[idx_star]
        Upair = np.zeros(len(r_vals))
        for i in range(len(coeff)):
            Upair += coeff[i]*Ucg.U_funcs[1][i](*xyz_traj.T)
        plt.plot(r_vals, Upair, color='k', lw=3, label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))

        if not ylims is None:
            plt.ylim(*ylims)

        plt.xlims(*xlims)

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
                    if not ylims is None:
                        ax.set_ylim(*ylims)
                    ax.set_xlim(*xlims)

                    if j == 0:
                        ax.set_ylabel(r"$U_{\mathrm{pair}}(r_{ij})$")
                    if i == (ncols - 1):
                        ax.set_xlabel(r"$r_{ij}$ (nm)")
                else:
                    sep = seps[pot_idx]
                    c_idx_start = pot_idx*n_pair_gauss

                    # plot for each regularization
                    for n in range(len(plot_idxs)):
                        alpha_idx = np.argwhere(vl_mse[plot_idxs[n],:,0] == vl_mse[plot_idxs[n],:,0].min())[:,0][0]
                        coeff = coeffs[plot_idxs[n], new_alpha_idx]
                        #coeff = coeffs[plot_idxs[n], alpha_idx]

                        Upair = np.zeros(len(r_vals))
                        Upair += f_mult_12[plot_idxs[n]]*Ucg.U_funcs[1][0](*xyz_traj.T)
                        for k in range(n_pair_gauss):
                            c_k = coeff[c_idx_start + k]
                            Upair += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

                        coeff_reg = coeffs[plot_idxs[n], alpha_reg_idx]
                        Upair_reg = np.zeros(len(r_vals))
                        Upair_reg += f_mult_12[plot_idxs[n]]*Ucg.U_funcs[1][0](*xyz_traj.T)
                        for k in range(n_pair_gauss):
                            c_k = coeff_reg[c_idx_start + k]
                            Upair_reg += c_k*Ucg.U_funcs[1][c_idx_start + k + 1](*xyz_traj.T)

                        if pot_idx == 0:
                            ax.plot(r_vals, Upair, color='k', lw=3, label=r"$\sigma^*={:.2f} \mathrm{{nm}}$  $\alpha^* = {:.1e}$".format(plot_sigs[n], sig_alphas[new_alpha_idx]))
                        else:
                            ax.plot(r_vals, Upair, color='k', lw=3)

                        #if pot_idx == 0:
                        #    if n == star_idx:
                        #        ax.plot(r_vals, Upair, color='k', lw=3, label=r"$\sigma^*={:.2f}$  $\alpha^* = {:.2e}$".format(plot_sigs[n], sig_alphas[alpha_idx]))
                        #        ax.plot(r_vals, Upair_reg, color='k', ls='--', lw=3, label=r"$\sigma^*={:.2f}$  $\alpha = {:.2e}$".format(plot_sigs[n], sig_alphas[alpha_reg_idx]))
                        #    else:
                        #        ln1 = ax.plot(r_vals, Upair, label=r"$\sigma ={:.2f}$  $\alpha^* = {:.2e}$".format(plot_sigs[n], sig_alphas[alpha_idx]))
                        #        ax.plot(r_vals, Upair_reg, color=ln1[0].get_color(), ls="--", label=r"$\sigma={:.2f}$  $\alpha = {:.2e}$".format(plot_sigs[n], sig_alphas[alpha_reg_idx]))
                        #else:
                        #    if n == star_idx:
                        #        ax.plot(r_vals, Upair, color='k', lw=3)
                        #        ax.plot(r_vals, Upair_reg, color='k', ls="--", lw=3)
                        #    else:
                        #        ln1 = ax.plot(r_vals, Upair)
                        #        ax.plot(r_vals, Upair_reg, ls="--", color=ln1[0].get_color())

                    if pot_idx == 0:
                        ax.legend()

                    if not ylims is None:
                        ax.set_ylim(*ylims)
                    ax.set_xlim(*xlims)

                    if j == 0:
                        ax.set_ylabel(r"$U_{\mathrm{pair}}(r_{ij})$")
                    if i == (ncols - 1):
                        ax.set_xlabel(r"$r_{ij}$ (nm)")

                    ax.annotate(r"$|i - j| = {:d}$".format(seps[pot_idx]), fontsize=16,
                            xy=(0,0), xytext=(0.55, 0.7), 
                            xycoords="axes fraction", textcoords="axes fraction")
                    #ax.annotate(r"$|i - j| = {:d}$".format(seps[pot_idx]), fontsize=16,
                    #        xy=(0,0), xytext=(0.55, 0.05), 
                    #        xycoords="axes fraction", textcoords="axes fraction")
        plt.savefig("{}compare_Upair_new_crit.pdf".format(prefix))
        plt.savefig("{}compare_Upair_new_crit.png".format(prefix))

    elif Ucg.pair_symmetry == "unique":
        raise NotImplementedError

        coord_idxs_by_seq_sep = Ucg._generate_pairwise_idxs(bond_cutoff=Ucg.bond_cutoff)

        fig, axes = plt.subplots(N, N, figsize=(4*N, 4*N), sharex=True)
        for i in range(N):
            for j in range(N):
                if j >= i + bcut:
                    # for seq sep |j - i|
                    ax = axes[i,j]

                    sep = np.abs(j - i)

                    # plot for each regularization
                    for n in range(len(idxs)):
                        coeff = coeffs[idxs[n]]
                        c_idx = 0

                        coeff[sep]
                        for i in range(len(coeff) - 1):
                            ax = axes[i, j]

                            Upair = np.zeros(len(r_vals))
                            # excluded volume
                            Upair += coeff[0]*Ucg.U_funcs[1][0](*xyz_traj.T)
                            for i in range(len(coeff)):
                                Upair += coeff[i]*Ucg.U_funcs[1][i](*xyz_traj.T)

                            #ax.plot(r_vals, Upair, label=r"$\alpha={:.2e}$".format(alphas[idxs[n]]))
                            ax.plot(r_vals, Upair)

                    coeff = coeffs[idx_star]
                    Upair = np.zeros(len(r_vals))
                    for i in range(len(coeff)):
                        Upair += coeff[i]*Ucg.U_funcs[1][i](*xyz_traj.T)
                    plt.plot(r_vals, Upair, color='k', lw=3, label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))

                    if not ylims is None:
                        ax.set_ylim(*ylims)
                    ax.set_xlim(*xlims)
                     
                else:
                    # remove axis junk to declutter
                    pass

    plt.savefig("{}compare_Upair.pdf".format(prefix))
    plt.savefig("{}compare_Upair.png".format(prefix))

def plot_Xcoeff_vs_d(idxs, idx_star, coeffs, alphas, X, d, prefix):

    plt.figure()
    for i in range(len(idxs)):
        dfit = np.dot(X, coeffs[idxs[i]])
        plt.plot((dfit - d)**2, '.', label=r"$\alpha={:.2e}$".format(alphas[idxs[i]]))

    dfit = np.dot(X, coeffs[idx_star])
    plt.plot((dfit - d)**2, 'k.', label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))
    plt.ylabel(r"$(\mathbf{X}c^* - \mathbf{d})^2$")
    plt.legend()
    plt.savefig("{}compare_d_residual.pdf".format(prefix))
    plt.savefig("{}compare_d_residual.png".format(prefix))

    plt.figure()
    plt.plot(d, 'ko', label="Target $\mathbf{d}$")
    for i in range(len(idxs)):
        dfit = np.dot(X, coeffs[idxs[i]])
        plt.plot(dfit, '.', label=r"$\alpha={:.2e}$".format(alphas[idxs[i]]))

    dfit = np.dot(X, coeffs[idx_star])
    plt.plot(dfit, '.', label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))
    plt.ylabel(r"$\mathbf{d}$  or  $\mathbf{X}c^*$ ")
    plt.legend()
    plt.savefig("{}compare_d_fit_target.pdf".format(prefix))
    plt.savefig("{}compare_d_fit_target.png".format(prefix))

def split_trajs_into_train_and_test_sets(trajnames, psinames, n_cross_val_sets=5):
    """Split trajs into sets with roughly same number of frames

    Trajectories are assigned into sets which will be used as training and test
    data sets. For simplicity they are assigned whole, so the number of frames
    in each set will vary.
    
    Parameters
    ----------
    trajnames : list, str
        List of trajectory filenames.

    total_n_frames : int, opt.
        Total number of trajectory frames.

    n_cross_val_sets : int, default=5
        Desired number of trajectory sets.
        
    """

    traj_n_frames = []
    for n in range(len(trajnames)):
        length = 0
        for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
            length += chunk.n_frames
        traj_n_frames.append(length)
    total_n_frames = sum(traj_n_frames)

    n_frames_in_set = total_n_frames/n_cross_val_sets

    traj_set = []
    psi_set = []
    traj_set_frames = []
    temp_traj_names = []
    temp_psi_names = []
    temp_frame_count = 0
    for n in range(len(traj_n_frames)):
        if temp_frame_count >= n_frames_in_set:
            # finish a set when it has desired number of frames
            traj_set.append(temp_traj_names)
            psi_set.append(temp_psi_names)
            traj_set_frames.append(temp_frame_count)

            # start over
            temp_traj_names = [trajnames[n]]
            temp_psi_names = [psinames[n]]
            temp_frame_count = traj_n_frames[n]
        else:
            temp_traj_names.append(trajnames[n])
            temp_psi_names.append(psinames[n])
            temp_frame_count += traj_n_frames[n]

        if n == len(traj_n_frames) - 1:
            traj_set.append(temp_traj_names)
            psi_set.append(temp_psi_names)
            traj_set_frames.append(temp_frame_count)

    with open("traj_sets_{}.txt".format(n_cross_val_sets), "w") as fout:
        for i in range(len(traj_set)):
            info_str = str(traj_set_frames[i])
            info_str += " " + " ".join(traj_set[i]) + "\n"
            fout.write(info_str)

    return traj_set, traj_set_frames, psi_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("msm_savedir", type=str)
    parser.add_argument("--psi_dims", type=int, default=1)
    parser.add_argument("--a_coeff", type=float, default=None)
    parser.add_argument("--using_cv", action="store_true")
    parser.add_argument("--n_basis", type=int, default=-1)
    parser.add_argument("--n_test", type=int, default=-1)
    parser.add_argument("--n_pair_gauss", type=int, default=-1)
    parser.add_argument("--pair_symmetry", type=str, default=None)
    parser.add_argument("--bond_cutoff", type=int, default=4)
    parser.add_argument("--lin_pot", action="store_true")
    parser.add_argument("--skip_trajs", action="store_true")
    parser.add_argument("--fix_back", action="store_true")
    parser.add_argument("--fix_exvol", action="store_true")
    parser.add_argument("--recalc_matrices", action="store_true")
    parser.add_argument("--noplot_ridge", action="store_true")
    args = parser.parse_args()

    msm_savedir = args.msm_savedir
    M = args.psi_dims
    a_coeff = args.a_coeff
    using_cv = args.using_cv
    n_cv_basis_funcs = args.n_basis
    n_cv_test_funcs = args.n_test
    n_pair_gauss = args.n_pair_gauss
    pair_symmetry = args.pair_symmetry
    bond_cutoff = args.bond_cutoff
    lin_pot = args.lin_pot
    skip_trajs = args.skip_trajs
    fix_back = args.fix_back
    fix_exvol = args.fix_exvol
    recalc_matrices = args.recalc_matrices
    noplot_ridge = args.noplot_ridge
    using_U0 = fix_back or fix_exvol

    print(" ".join(sys.argv))

    if (n_cv_basis_funcs != -1):
        print("Since n_basis ({}) is specified -> using_cv=True".format(n_cv_basis_funcs))
        using_cv = True
    else:
        if using_cv: 
            raise ValueError("Please specify n_test and n_basis")

    if n_pair_gauss != -1:
        if not pair_symmetry in ["shared", "seq_sep", "unique"]:
            raise ValueError("Must specificy pair_symmetry")

    #python ~/code/implicit_force_field/eigenpair_soln.py msm_dists --psi_dims 1 --using_cv --n_basis 40 --n_test 100 --fix_back --bond_cutoff 3 --a_coeff 0.027

    # If simulation were done with OpenMM gamma = 1 ps^-1 
    # then noise coefficient a_coeff = 1ps/mass_ply
    # a_coeff = 1/37 = 0.027

    n_beads = 25
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)

    if a_coeff is None:
        fixed_a = False
    else:
        fixed_a = True

    using_D2 = False
    n_cross_val_sets = 5

    cg_savedir = util.Ucg_dirname("eigenpair", M, using_U0, fix_back,
            fix_exvol, bond_cutoff, using_cv,
            n_cv_basis_funcs=n_cv_basis_funcs, n_cv_test_funcs=n_cv_test_funcs,
            a_coeff=a_coeff, n_pair_gauss=n_pair_gauss,
            pair_symmetry=pair_symmetry)

    print(cg_savedir)


    # create potential energy function
    Ucg, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(
            msm_savedir, n_beads, M, beta, fix_back, fix_exvol, using_cv,
            using_D2, n_cv_basis_funcs, n_cv_test_funcs, n_pair_gauss,
            bond_cutoff, a_coeff=a_coeff, pair_symmetry=pair_symmetry)

    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd") 
    ti_file = msm_savedir + "/tica_ti_ps.npy"
    psinames = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        temp_names = []
        for n in range(M):
            temp_names.append(msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, n+1))
        psinames.append(temp_names)

    if not os.path.exists(cg_savedir):
        os.mkdir(cg_savedir)

    ##################################################################
    # calculate matrix X and d 
    ##################################################################
    s_loss = loss.LinearSpectralLoss(topfile, trajnames, cg_savedir, n_cv_sets=n_cross_val_sets, recalc=recalc_matrices)

    if skip_trajs:
        include_trajs = [ x for x in range(1,len(trajnames), 4) ]
    else:
        include_trajs = [ x for x in range(len(trajnames)) ]

    if not s_loss.matrix_files_exist() or recalc_matrices:
        s_loss.assign_crossval_sets()
        s_loss.calc_matrices(Ucg, psinames, ti_file, M=M, coll_var_names=psinames, verbose=True, include_trajs=include_trajs)

    os.chdir(cg_savedir)

    rdg_alphas = np.logspace(-10, 8, 500)
    if not os.path.exists("rdg_valid_mse.npy") or recalc_matrices:
        print("Ridge regularization...")
        s_loss.solve(rdg_alphas)

        np.save("rdg_cstar.npy", s_loss.coeff_star)
        np.save("rdg_coeffs.npy", s_loss.coeffs)
        np.save("rdg_train_mse.npy", s_loss.train_mse)
        np.save("rdg_valid_mse.npy", s_loss.valid_mse)

        rdg_cstar = s_loss.coeff_star
        rdg_coeffs = s_loss.coeffs
        rdg_train_mse = s_loss.train_mse
        rdg_valid_mse = s_loss.valid_mse
    else:
        rdg_cstar = np.load("rdg_cstar.npy")
        rdg_coeffs = np.load("rdg_coeffs.npy")
        rdg_train_mse = np.load("rdg_train_mse.npy")
        rdg_valid_mse = np.load("rdg_valid_mse.npy")

    alpha_star_idx = np.argmin(rdg_valid_mse[:,0])
    if not noplot_ridge:
        print("Plotting ridge results...")
        iff.util.plot_train_test_mse(rdg_alphas, rdg_train_mse, rdg_valid_mse, 
                xlabel=r"Regularization $\alpha$", 
                ylabel="Mean squared error (MSE)", 
                title="Ridge regression", prefix="ridge_")


    if using_cv:
        print("Plotting Ucv...")
        rdg_idxs = [5, 50, 200, 300]
        plot_Ucg_vs_alpha(rdg_idxs, alpha_star_idx, rdg_coeffs, rdg_alphas, Ucg, cv_r0_basis, "rdg_", fixed_a=fixed_a)
    else:
        # cross validation as a function of sigma and alpha
        if not os.path.exists("rdg_fixed_sigma_coeffs.npy"):
            print("Cross-validating sigma_ex...")
            f_mult_12, sig_alphas, sig_coeffs, sig_tr_mse, sig_vl_mse = iff.util.scan_with_fixed_sigma(s_loss, Ucg, cv_r0_basis)
            sigma_idx, alpha_idx = np.argwhere(sig_vl_mse[:,:,0] == sig_vl_mse[:,:,0].min())[0]


            sig_cstar = np.concatenate([ np.array([f_mult_12[sigma_idx]]), sig_coeffs[sigma_idx, alpha_idx]])

            np.save("rdg_fixed_sigma_f_mult_12.npy", f_mult_12)
            np.save("rdg_fixed_sigma_alphas.npy", sig_alphas)
            np.save("rdg_fixed_sigma_cstar.npy", sig_cstar)
            np.save("rdg_fixed_sigma_coeffs.npy", sig_coeffs)
            np.save("rdg_fixed_sigma_train_mse.npy", sig_tr_mse)
            np.save("rdg_fixed_sigma_valid_mse.npy", sig_vl_mse)
        else:
            f_mult_12 = np.load("rdg_fixed_sigma_f_mult_12.npy")
            sig_alphas = np.load("rdg_fixed_sigma_alphas.npy")
            sig_cstar = np.load("rdg_fixed_sigma_cstar.npy")
            sig_coeffs = np.load("rdg_fixed_sigma_coeffs.npy")
            sig_tr_mse = np.load("rdg_fixed_sigma_train_mse.npy")
            sig_vl_mse = np.load("rdg_fixed_sigma_valid_mse.npy")

        # plot best potential for each value of sigma
        scaled_sigma = 0.373*(f_mult_12**(1./12))
        #r_vals = np.linspace(0.2, 1.5, 200)
        r_vals = np.linspace(0.1, 1.2, 200)

        sigma_orig = 0.373
        # Accept all solutions within 10% of the minimum validation error
        min_vl_idxs = np.argwhere(sig_vl_mse[:,:,0] <= 1.1*sig_vl_mse[:,:,0].min())

        # choose sigma closest to that of original simulation
        # choose largest alpha among those sigma
        new_sigma_idx = np.argmin((sigma_orig - scaled_sigma[np.unique(min_vl_idxs[:,0])])**2)
        new_alpha_idx = np.max((min_vl_idxs[min_vl_idxs[:,0] == new_sigma_idx,:])[:,1])

        sigma_star = scaled_sigma[new_sigma_idx]
        alpha_star = sig_alphas[new_alpha_idx]
        coeff_star = np.concatenate([ np.array([f_mult_12[new_sigma_idx]]), sig_coeffs[new_sigma_idx, new_alpha_idx]])

        print("Plotting Upair...")
        #plot_Upair_vs_alpha(rdg_idxs, alpha_star_idx, rdg_coeffs, rdg_alphas, Ucg, r_vals, n_pair_gauss, "rdg_", ylims=(-10, 10))
        #plot_Upair_for_best_sigma(Ucg, r_vals, scaled_sigma, f_mult_12, sig_coeffs, sig_vl_mse, n_pair_gauss, "sig_", ylims=(-2, 5))
        plot_Upair_for_best_sigma(Ucg, r_vals, coeff_star, sigma_star, alpha_star, n_pair_gauss, "sig_", ylims=(-2, 5))

        cstar = sig_cstar
    
    raise SystemExit
    os.chdir("..")
    s_loss.scalar_product_Gen_fj(Ucg, cstar, psinames, cv_names=psinames)
    os.chdir(cg_savedir)

    np.save("psi_fj.npy", s_loss.psi_fj)
    np.save("psi_gU0_fj.npy", s_loss.psi_gU0_fj)
    np.save("psi_gU1_fj.npy", s_loss.psi_gU1_fj)
    np.save("psi_Lap_fj.npy", s_loss.psi_Lap_fj)
    np.save("psi_Gen_fj.npy", s_loss.psi_Gen_fj)

