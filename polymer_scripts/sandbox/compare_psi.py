import os
import sys
import glob
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

import simtk.unit as unit
import simtk.openmm.app as app

import mdtraj as md

import simulation.openmm as sop
import implicit_force_field as iff

if __name__ == "__main__":
    """Plots TICA coordinates from my code against TICA coordinates saved by Pyemma"""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_idx", type=int)
    parser.add_argument("n_gauss", type=int)
    args = parser.parse_args()

    run_idx = args.run_idx
    n_gauss = args.n_gauss

    n_beads = 25
    #n_beads = 5
    #n_beads = 4
    #n_beads = 3
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)

    traj_idx = 1
    #topname = "c25_nosolv_min.pdb"
    topname = name + "_min.pdb"
    min_name = name + "_min_{}.pdb".format(traj_idx)
    log_name = name + "_{}.log".format(traj_idx)
    traj_name = name + "_traj_cent_{}.dcd".format(traj_idx)
    lastframe_name = name + "_fin_{}.pdb".format(traj_idx)


    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    r0, kb, theta0, ka = bonded_params
    app.element.polymer = app.element.Element(200, "Polymer", "Pl", mass_ply)

    sigma_ply_nm = sigma_ply/unit.nanometer
    #r0_wca_nm = sigma_ply_nm*(2**(1./6))
    eps_ply_kj = eps_ply/unit.kilojoule_per_mole
    kb_kj = kb/(unit.kilojoule_per_mole/(unit.nanometer**2))
    ka_kj = (ka/(unit.kilojoule_per_mole/(unit.radian**2)))
    theta0_rad = theta0/unit.radian
    r0_nm = r0/unit.nanometer

    #gauss_r0_nm = np.linspace(0.3, 1, 10)
    #gauss_w_nm = 0.1*np.ones(len(gauss_r0_nm))

    gauss_r0_nm = np.linspace(0.3, 1, n_gauss)
    gauss_sigma = gauss_r0_nm[1] - gauss_r0_nm[0]
    gauss_w_nm = gauss_sigma*np.ones(len(gauss_r0_nm))

    #msm_savedir = "msm_dih_dists"
    msm_savedir = "msm_dists"

    #M = 3
    M = 1   # number of eigenvectors to use

    #cg_savedir = "Ucg_eigenpair_fixed_bonds_angles_free_pairs_10"
    cg_savedir = "Ucg_eigenpair_fixed_bonds_angles_free_pairs_{}_CV_{}".format(n_gauss, M)

    # scaling the terms of the potential reduces the condition number of the
    # force matrix by several orders of magnitude. Pre-conditioning by column
    # multiplication

    print "creating Ucg..."
    # coarse-grain polymer potential with free parameters
    Ucg = iff.basis_library.PolymerModel(n_beads)
    Ucg._assign_harmonic_bonds(r0_nm, scale_factor=kb_kj, fixed=True)
    Ucg._assign_harmonic_angles(theta0_rad, scale_factor=ka_kj, fixed=True)
    #Ucg._assign_LJ6(sigma_ply_nm, scale_factor=eps_ply_kj)
    Ucg._assign_inverse_r12(sigma_ply_nm, scale_factor=0.5, fixed=True)
    Ucg._assign_pairwise_gaussians(gauss_r0_nm, gauss_w_nm, scale_factor=10)

    ply_idxs = np.arange(25)
    pair_idxs = []
    for i in range(len(ply_idxs) - 1):
        for j in range(i + 4, len(ply_idxs)):
            pair_idxs.append([ply_idxs[i], ply_idxs[j]])
    pair_idxs = np.array(pair_idxs)

    # add test functions
    use_cv_f_j = True
    if use_cv_f_j:
        # centers of test functions in collective variable (CV) space
        cv_r0 = np.load(msm_savedir + "/psi1_mid_bin.npy")
        cv_w = np.abs(cv_r0[1] - cv_r0[0])*np.ones(len(cv_r0), float)
        #cv_r0 = np.array([ [cv_r0[i]] for i in range(len(cv_r0)) ])
        cv_r0 = cv_r0.reshape((len(cv_r0),1))

        #cv_coeff = np.array([ [x] for x in np.load(msm_savedir + "/tica_eigenvects.npy")[:,0]])
        cv_coeff = np.load(msm_savedir + "/tica_eigenvects.npy")[:,:M]
        cv_mean = np.load(msm_savedir + "/tica_mean.npy")

        # TODO: add additional features
        Ucg.linear_collective_variables(["dist"], pair_idxs, cv_coeff, cv_mean)
        Ucg.gaussian_cv_test_funcs(cv_r0, cv_w)
    else:
        Ucg._assign_bond_funcs([r0_nm], [0.3])
        Ucg._assign_angle_funcs([theta0_rad], [4])
        Ucg._assign_pairwise_funcs(gauss_r0_nm, gauss_w_nm)

    D = Ucg.n_dof           # number of degrees of freedom
    R = Ucg.n_params        # number of free model parameters
    if use_cv_f_j:
        P = Ucg.n_test_funcs_cv # number of test functions
    else:
        P = Ucg.n_test_funcs    # number of test functions

    ##########################################################
    # calculate integrated sindy (eigenpair) matrix equation.
    ########################################################## 
    # load tics
    topfile = glob.glob("run_{}/".format(run_idx) + name + "_min_cent.pdb")[0]
    trajnames = glob.glob("run_{}/".format(run_idx) + name + "_traj_cent_*.dcd") 
    traj_idxs = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        traj_idxs.append([idx1, idx2])

    kappa = 1./np.load(msm_savedir + "/tica_ti.npy")[:M]

    X = np.zeros((M*P, R+1), float)
    d = np.zeros(M*P, float)

    print "calculating matrix elements..."
    Ntot = 0
    for n in range(len(traj_idxs)):
        starttime = time.time()

        print "traj: ", n+1
        sys.stdout.flush()
        idx1, idx2 = traj_idxs[n]

        # load tics
        psi_traj = []
        for i in range(M):
            # save TIC with indices of corresponding traj
            tic_saveas = msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, i+1)
            psi_traj.append(np.load(tic_saveas))
        psi_traj = np.array(psi_traj).T

        if len(psi_traj.shape) == 1:
            psi_traj = psi_traj.reshape((psi_traj.shape[0], 1))

        # calculate psi
        chunk_num = 1
        psi_calc = []
        for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
            print "    chunk: ", chunk_num
            sys.stdout.flush()
            chunk_num += 1
            psi_calc.append(Ucg._collective_variable_value(chunk))
        psi_calc = np.concatenate(psi_calc)

        min_calc_traj = (time.time() - starttime)
        print "calculation took: {:.4f} sec".format(min_calc_traj)
        sys.stdout.flush()

    x, y = psi_traj[:,0], psi_calc[:,0]
    corr = np.corrcoef(x,y)[0,1]
    ymin, ymax = -1, 1.8
    plt.figure()
    plt.plot([ymin, ymax], [ymin, ymax], 'k--')
    plt.plot(x, y, '.')
    plt.xlabel(r"True $\psi_1$")
    plt.ylabel(r"Calc $\psi_1$")
    plt.title("Corr = {:.4f}".format(corr))
    plt.savefig(msm_savedir + "/psi_true_vs_calc.pdf")
    plt.savefig(msm_savedir + "/psi_true_vs_calc.png")

