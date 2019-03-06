from __future__ import print_function, absolute_import
import os
import time
import sys
import glob
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

def plot_Ucg_vs_alpha(idxs, idx_star, coeffs, alphas, Ucg, cv_r0, prefix, ylim=None, fixed_a=False):

    plt.figure()
    for n in range(len(idxs)):
        coeff = coeffs[idxs[n]]
        U = np.zeros(len(cv_r0))
        for i in range(len(Ucg.cv_U_funcs)):
            U += coeff[Ucg.n_cart_params + i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
        U -= U.min()

        plt.plot(cv_r0[:,0], U, label=r"$\alpha={:.2e}$".format(alphas[idxs[n]]))

    coeff = coeffs[idx_star]
    U = np.zeros(len(cv_r0))
    for i in range(len(Ucg.cv_U_funcs)):
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

def plot_Ucg_vs_time():
    coeff = np.load(cg_savedir + "/rdg_cstar.npy")
    U_all = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]

        U_temp = []
        for chunk in md.iterload(tname, top=topfile):
            xyz_traj = np.reshape(chunk.xyz, (-1, 75))
            cv_traj = Ucg.calculate_cv(xyz_traj)
            U_chunk = np.einsum("k,tk->t", coeff, Ucg.potential_U1(xyz_traj, cv_traj))
            U_temp.append(U_chunk)
        U_temp = np.concatenate(U_temp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("msm_savedir", type=str)
    parser.add_argument("--psi_dims", type=int, default=1)
    parser.add_argument("--using_cv", action="store_true")
    parser.add_argument("--n_basis", type=int, default=-1)
    parser.add_argument("--n_test", type=int, default=-1)
    parser.add_argument("--n_pair_gauss", type=int, default=-1)
    parser.add_argument("--pair_symmetry", type=str, default=None)
    parser.add_argument("--bond_cutoff", type=int, default=4)
    parser.add_argument("--fix_back", action="store_true")
    parser.add_argument("--fix_exvol", action="store_true")
    parser.add_argument("--recalc_matrices", action="store_true")
    args = parser.parse_args()

    msm_savedir = args.msm_savedir
    M = args.psi_dims
    using_cv = args.using_cv
    n_cv_basis_funcs = args.n_basis
    n_cv_test_funcs = args.n_test
    n_pair_gauss = args.n_pair_gauss
    bond_cutoff = args.bond_cutoff
    fix_back = args.fix_back
    fix_exvol = args.fix_exvol
    recalc_matrices = args.recalc_matrices
    using_U0 = fix_back or fix_exvol

    print(" ".join(sys.argv))

    if (n_cv_basis_funcs != -1):
        print("Since n_test ({}) and n_basis ({}) are specified -> using_cv=True".format(n_cv_test_funcs, n_cv_basis_funcs))
        using_cv = True
    else:
        if using_cv:
            raise ValueError("Please specify n_test and n_basis")

    if n_pair_gauss != -1:
        if not pair_symmetry in ["shared", "seq_sep", "unique"]:
            raise ValueError("Must specificy pair_symmetry")

    #python ~/code/implicit_force_field/force_matching.py msm_dists --psi_dims 1 --n_basis 40 --n_test 100 --fixed_bonds
    #python ~/code/implicit_force_field/force_matching.py msm_dists --psi_dims 1 --n_basis 40 --n_test 100 --bond_cutoff 4

    n_beads = 25
    n_dim = 3*n_beads
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)
    #n_pair_gauss = 10

    using_D2 = False
    n_cross_val_sets = 5

    cg_savedir = util.Ucg_dirname("force-matching", M, using_U0, fix_back,
            fix_exvol, bond_cutoff, using_cv, n_cv_basis_funcs=n_cv_basis_funcs,
            n_cv_test_funcs=n_cv_test_funcs, n_pair_gauss=n_pair_gauss,
            pair_symmetry=pair_symmetry)

    print(cg_savedir)

    #print("building basis function database...")
    Ucg, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(
            msm_savedir, n_beads, M, beta, fix_back, fix_exvol, using_cv,
            using_D2, n_cv_basis_funcs, n_cv_test_funcs, n_pair_gauss,
            bond_cutoff, pair_symmetry=pair_symmetry)

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

    #traj_frames = []
    #for i in range(len(trajnames)):
    #    length = 0
    #    for chunk in md.iterload(trajnames[i], top=topfile):
    #        length += chunk.n_frames
    #    traj_frames.append(length)

    if not os.path.exists(cg_savedir):
        os.mkdir(cg_savedir)

    ##################################################################
    # calculate matrix X and d 
    ##################################################################
    loss_func = loss.LinearForceMatchingLoss(topfile, trajnames, cg_savedir, n_cv_sets=n_cross_val_sets, recalc=recalc_matrices)

    if not loss_func.matrix_files_exist() or recalc_matrices:
        loss_func.assign_crossval_sets()
        loss_func.calc_matrices(Ucg, forcenames, coll_var_names=psinames, verbose=True)

    os.chdir(cg_savedir)

    if not os.path.exists("rdg_valid_mse.npy"):
        rdg_alphas = np.logspace(-10, 8, 500)
        loss_func.solve(rdg_alphas)

        np.save("rdg_cstar.npy", loss_func.coeff_star)
        np.save("rdg_coeffs.npy", loss_func.coeffs)
        np.save("rdg_train_mse.npy", loss_func.train_mse)
        np.save("rdg_valid_mse.npy", loss_func.valid_mse)

        rdg_idxs = [5, 50, 200, 300]
        plot_Ucg_vs_alpha(rdg_idxs, loss_func.alpha_star_idx, loss_func.coeffs, rdg_alphas, Ucg, cv_r0_basis, "rdg_")
        #plot_Xcoeff_vs_d(rdg_idxs, rdg_idx_star, rdg_coeffs, rdg_alphas, loss_func.X, loss_func.d, "rdg_")

        iff.util.plot_train_test_mse(rdg_alphas, loss_func.train_mse, loss_func.valid_mse, 
                xlabel=r"Regularization $\alpha$", 
                ylabel="Mean squared error (MSE)", 
                title="Ridge regression", prefix="ridge_")

    if not os.path.exists("rdg_fixed_sigma_cstar.npy") or True:
        f_mult_12, alphas, all_coeffs, tr_mse, vl_mse = iff.util.scan_with_fixed_sigma(loss_func, Ucg, cv_r0_basis)
        sigma_idx, alpha_idx = np.argwhere(vl_mse[:,:,0] == vl_mse[:,:,0].min())[0]
        new_coeffs = np.concatenate([ np.array([f_mult_12[sigma_idx]]), all_coeffs[sigma_idx, alpha_idx]])
        np.save("rdg_fixed_sigma_cstar.npy", new_coeffs)
    else:
        new_coeffs = np.load("rdg_fixed_sigma_cstar.npy")

