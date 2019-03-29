from __future__ import print_function, absolute_import
import os
import glob
import time
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
    parser.add_argument("cg_method", type=str)
    parser.add_argument("--psi_dims", type=int, default=1)
    parser.add_argument("--a_coeff", type=float, default=None)
    parser.add_argument("--using_cv", action="store_true")
    parser.add_argument("--n_basis", type=int, default=-1)
    parser.add_argument("--n_test", type=int, default=-1)
    parser.add_argument("--n_pair_gauss", type=int, default=-1)
    parser.add_argument("--pair_symmetry", type=str, default=None)
    parser.add_argument("--bond_cutoff", type=int, default=4)
    parser.add_argument("--lin_pot", action="store_true")
    parser.add_argument("--alpha_lims", nargs=3, type=float, default=[-10, 2, 100])
    parser.add_argument("--skip_trajs", type=int, default=1)
    parser.add_argument("--save_by_traj", action="store_true")
    parser.add_argument("--fix_back", action="store_true")
    parser.add_argument("--fix_exvol", action="store_true")
    parser.add_argument("--recalc_matrices", action="store_true")
    parser.add_argument("--recalc_cross_val", action="store_true")
    parser.add_argument("--plot_scalar", action="store_true")
    parser.add_argument("--recalc_scalar", action="store_true")
    parser.add_argument("--n_fixed_sigma", type=int, default=100)
    args = parser.parse_args()

    msm_savedir = args.msm_savedir
    cg_method = args.cg_method
    M = args.psi_dims
    a_coeff = args.a_coeff
    using_cv = args.using_cv
    n_cv_basis_funcs = args.n_basis
    n_cv_test_funcs = args.n_test
    n_pair_gauss = args.n_pair_gauss
    pair_symmetry = args.pair_symmetry
    bond_cutoff = args.bond_cutoff
    lin_pot = args.lin_pot
    alpha_lims = args.alpha_lims
    skip_trajs = args.skip_trajs
    save_by_traj = args.save_by_traj
    fix_back = args.fix_back
    fix_exvol = args.fix_exvol
    recalc_matrices = args.recalc_matrices
    recalc_cross_val = args.recalc_cross_val
    plot_scalar = args.plot_scalar
    recalc_scalar = args.recalc_scalar
    n_fixed_sigma = args.n_fixed_sigma
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

    cg_savedir = util.test_Ucg_dirname(cg_method, M, using_U0, fix_back,
            fix_exvol, bond_cutoff, using_cv,
            n_cv_basis_funcs=n_cv_basis_funcs, n_cv_test_funcs=n_cv_test_funcs,
            a_coeff=a_coeff, n_pair_gauss=n_pair_gauss, cv_lin_pot=lin_pot,
            pair_symmetry=pair_symmetry)

    print(cg_savedir)

    # create potential energy function
    Ucg, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg( msm_savedir,
            n_beads, M, beta, fix_back, fix_exvol, using_cv, using_D2,
            n_cv_basis_funcs, n_cv_test_funcs, n_pair_gauss, bond_cutoff,
            cv_lin_pot=lin_pot, a_coeff=a_coeff, pair_symmetry=pair_symmetry)

    if cg_method == "force-matching":
        # only get trajectories that have saved forces
        temp_forcenames = glob.glob("run_*/" + name + "_forces_*.dat") 

        forcenames = []
        current_time = time.time()
        for i in range(len(temp_forcenames)):
            min_since_mod = np.abs(current_time - os.path.getmtime(temp_forcenames[i]))/60.
            if min_since_mod > 10:
                forcenames.append(temp_forcenames[i])
            else:
                print("skipping: " + temp_forcenames[i])

        topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
        rundirs = []
        psinames = []
        trajnames = []
        for i in range(len(forcenames)):
            fname = forcenames[i]
            idx1 = (os.path.dirname(fname)).split("_")[-1]
            idx2 = (os.path.basename(fname)).split(".dat")[0].split("_")[-1]

            traj_name = "run_{}/{}_traj_cent_{}.dcd".format(idx1, name, idx2)
            if not os.path.exists(traj_name):
                #raise ValueError("Trajectory does not exist: " + traj_name)
                print("Trajectory does not exist: " + traj_name)

            trajnames.append(traj_name)

            temp_names = []
            for n in range(M):
                psi_name = msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, n+1)
                if not os.path.exists(psi_name):
                    psi_temp = []
                    for chunk in md.iterload(trajnames[i], top=topfile):
                        xyz_traj = np.reshape(chunk.xyz, (-1, 75))
                        psi_chunk = Ucg.calculate_cv(xyz_traj)
                        psi_temp.append(psi_chunk[:,n])
                    psi_temp = np.concatenate(psi_temp)
                    np.save(psi_name, psi_temp)
                temp_names.append(psi_name)
            psinames.append(temp_names)
    else:
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

    include_trajs = [ x for x in range(0, len(trajnames), skip_trajs) ]

    if not os.path.exists(cg_savedir):
        os.mkdir(cg_savedir)

    # choose loss function for coarse-graining method
    if cg_method == "force-matching":
        loss_func = loss.LinearForceMatchingLoss(topfile, trajnames,
                cg_savedir, n_cv_sets=n_cross_val_sets, recalc=recalc_matrices,
                save_by_traj=save_by_traj)

        if not loss_func.matrix_files_exist() or recalc_matrices:
            loss_func.assign_crossval_sets()
            loss_func.calc_matrices(Ucg, forcenames, coll_var_names=psinames, verbose=True, include_trajs=include_trajs, chunksize=100)
    elif cg_method == "eigenpair":
        loss_func = loss.LinearSpectralLoss(topfile, trajnames, cg_savedir,
                n_cv_sets=n_cross_val_sets, recalc=recalc_matrices,
                save_by_traj=save_by_traj)

        if not loss_func.matrix_files_exist() or recalc_matrices:
            loss_func.assign_crossval_sets()
            loss_func.calc_matrices(Ucg, psinames, ti_file, M=M, coll_var_names=psinames, verbose=True, include_trajs=include_trajs)

    os.chdir(cg_savedir)

    # plot cross validated solutions
    if not fix_exvol:
        # k-fold cross validation with respect to both sigma and alpha
        if not os.path.exists("rdg_fixed_sigma_coeffs.npy") or recalc_cross_val:
            print("Cross-validating sigma_ex...")
            f_mult_12, sig_alphas, sig_coeffs, sig_tr_mse, sig_vl_mse = iff.util.scan_with_fixed_sigma(loss_func, Ucg, cv_r0_basis, alpha_lims, n_fixed_sigma=n_fixed_sigma)
            #sigma_idx, alpha_idx = np.argwhere(sig_vl_mse[:,:,0] == sig_vl_mse[:,:,0].min())[0]

            np.save("rdg_fixed_sigma_f_mult_12.npy", f_mult_12)
            np.save("rdg_fixed_sigma_alphas.npy", sig_alphas)
            np.save("rdg_fixed_sigma_coeffs.npy", sig_coeffs)
            np.save("rdg_fixed_sigma_train_mse.npy", sig_tr_mse)
            np.save("rdg_fixed_sigma_valid_mse.npy", sig_vl_mse)
        else:
            f_mult_12 = np.load("rdg_fixed_sigma_f_mult_12.npy")
            sig_alphas = np.load("rdg_fixed_sigma_alphas.npy")
            sig_coeffs = np.load("rdg_fixed_sigma_coeffs.npy")
            sig_tr_mse = np.load("rdg_fixed_sigma_train_mse.npy")
            sig_vl_mse = np.load("rdg_fixed_sigma_valid_mse.npy")

        print("Plotting cross-val vs (sigma, alpha)...")
        sigma_orig = 0.373
        scaled_sigma = sigma_orig*(f_mult_12**(1./12))
        sigma_orig_idx = np.argmin((sigma_orig - scaled_sigma)**2)

        # check if original sigma is within acceptable range to global minimum
        sig_idx, alp_idx = np.argwhere(sig_vl_mse[:,:,0] == sig_vl_mse[:,:,0].min())[0]
        acc_idxs = np.argwhere(sig_vl_mse[:,:,0] <= (sig_vl_mse[sig_idx, alp_idx, 0] + 0.5*sig_vl_mse[sig_idx, alp_idx, 0]))

        if sigma_orig_idx in np.unique(acc_idxs[:,0]):
            # choose original sigma 
            optimal_sigma_idx = sigma_orig_idx
            optimal_alpha_idx = np.argmin(sig_vl_mse[optimal_sigma_idx,:,0])
        else:
            # original radius is not in the acceptable domain
            optimal_sigma_idx = sig_idx
            optimal_alpha_idx = alp_idx

        opt_std = sig_vl_mse[optimal_sigma_idx, optimal_alpha_idx, 1] 

        min_vl_domain = sig_vl_mse[:,:,0] <= (sig_vl_mse[optimal_sigma_idx, optimal_alpha_idx, 0] + opt_std)
        min_vl_idxs = np.argwhere(min_vl_domain)

        max_alpha_idx = np.max((min_vl_idxs[min_vl_idxs[:,0] == optimal_sigma_idx,:])[:,1])
        min_alpha_idx = np.min((min_vl_idxs[min_vl_idxs[:,0] == optimal_sigma_idx,:])[:,1])

        sigma_star = scaled_sigma[optimal_sigma_idx]
        alpha_star = sig_alphas[optimal_alpha_idx]
        alpha_max = sig_alphas[max_alpha_idx]
        alpha_min = sig_alphas[min_alpha_idx]

        coeff_star = np.concatenate([ np.array([f_mult_12[optimal_sigma_idx]]), sig_coeffs[optimal_sigma_idx, optimal_alpha_idx]])
        coeff_max = np.concatenate([ np.array([f_mult_12[optimal_sigma_idx]]), sig_coeffs[optimal_sigma_idx, max_alpha_idx]])
        coeff_min = np.concatenate([ np.array([f_mult_12[optimal_sigma_idx]]), sig_coeffs[optimal_sigma_idx, min_alpha_idx]])

        np.save("rdg_fixed_sigma_cstar.npy", coeff_star)

        # plot cross validation score with respect to sigma and alpha, add contour at
        # 10% of minimum. Add marker at the chosen values.
        cont_lvl = sig_vl_mse[optimal_sigma_idx, optimal_alpha_idx, 0] + opt_std 

        X, Y = np.meshgrid(scaled_sigma, sig_alphas)
        plt.figure()
        pcol = plt.pcolormesh(X, Y, np.log10(sig_vl_mse[:,:,0]).T, linewidth=0, rasterized=True)
        pcol.set_edgecolor("face")
        plt.contour(X, Y, sig_vl_mse[:,:,0].T, [cont_lvl], colors="k", linewidths=3)
        plt.plot([sigma_star], [alpha_max], markersize=10, marker="o", color="w")
        plt.plot([sigma_star], [alpha_min], markersize=10, marker="o", color="w")
        plt.plot([sigma_star], [alpha_star], markersize=14, marker="*", color="y")
        plt.xlabel(r"Scaled radius $\sigma'$ (nm)")
        plt.ylabel(r"Regularization $\alpha$")
        plt.semilogy()
        cbar = plt.colorbar(mappable=pcol)
        cbar.set_label("log(crossval score)")
        plt.savefig("cross_val_vs_sigma_alpha_contour.pdf")
        plt.savefig("cross_val_vs_sigma_alpha_contour.png")

        plot_cffs = (coeff_star, coeff_min, coeff_max)
        plot_alph = (alpha_star, alpha_min, alpha_max)

        if using_cv:
            print("Plotting Ucv...")
            cv_vals = np.linspace(1.3*cv_r0_basis.min(), 1.2*cv_r0_basis.max(), 200)
            iff.util.plot_Ucv_for_best_sigma(Ucg, cv_vals, plot_cffs, plot_alph, sigma_star, "sig_", ylims=(-10, 100))
        else:
            print("Plotting Upair...")
            r_vals = np.linspace(0.1, 1.2, 200)
            iff.util.plot_Upair_for_best_sigma(Ucg, r_vals, plot_cffs, plot_alph, sigma_star, n_pair_gauss, "sig_", ylims=(-2, 5), with_min=True)
    else:
        # cross validation with respect to regularization parameter alpha
        rdg_alphas = np.logspace(-10, 8, 500)
        if not os.path.exists("rdg_valid_mse.npy") or recalc_cross_val:
            print("Ridge regularization...")
            loss_func.solve(rdg_alphas)

            np.save("rdg_cstar.npy", loss_func.coeff_star)
            np.save("rdg_coeffs.npy", loss_func.coeffs)
            np.save("rdg_train_mse.npy", loss_func.train_mse)
            np.save("rdg_valid_mse.npy", loss_func.valid_mse)

            rdg_cstar = loss_func.coeff_star
            rdg_coeffs = loss_func.coeffs
            rdg_train_mse = loss_func.train_mse
            rdg_valid_mse = loss_func.valid_mse
        else:
            rdg_cstar = np.load("rdg_cstar.npy")
            rdg_coeffs = np.load("rdg_coeffs.npy")
            rdg_train_mse = np.load("rdg_train_mse.npy")
            rdg_valid_mse = np.load("rdg_valid_mse.npy")

        alpha_max_idx = np.argmin(rdg_valid_mse[:,0])

        print("Plotting ridge results...")
        iff.util.plot_train_test_mse(rdg_alphas, rdg_train_mse, rdg_valid_mse, 
                xlabel=r"Regularization $\alpha$", 
                ylabel="Mean squared error (MSE)", 
                title="Ridge regression", prefix="ridge_")
    
    if plot_scalar:
        files_exist = [ os.path.exists(fname) for fname in ["psi_fj.npy",
            "psi_gU0_fj.npy", "psi_gU1_fj.npy", "psi_Lap_fj.npy", "psi_Gen_fj.npy"]]
        if not np.all(files_exist) or recalc_scalar:
            os.chdir("..")
            #include_trajs = np.arange(0, len(trajnames), 10)
            #include_trajs = np.arange(0, 10)
            include_trajs = np.arange(0, len(trajnames))
            loss_func.scalar_product_Gen_fj(Ucg, coeff_star, psinames, cv_names=psinames, include_trajs=include_trajs)
            os.chdir(cg_savedir)

            np.save("psi_fj.npy", loss_func.psi_fj)
            np.save("psi_gU0_fj.npy", loss_func.psi_gU0_fj)
            np.save("psi_gU1_fj.npy", loss_func.psi_gU1_fj)
            np.save("psi_Lap_fj.npy", loss_func.psi_Lap_fj)
            np.save("psi_Gen_fj.npy", loss_func.psi_Gen_fj)

        import implicit_force_field.polymer_scripts.plot_scalar
        implicit_force_field.polymer_scripts.plot_scalar.plot_scalar()
