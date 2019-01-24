from __future__ import print_function
import os
import shutil
import time
import argparse
import numpy as np
import scipy.interpolate

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import mdtraj as md

import simulation.openmm as sop

import implicit_force_field.polymer_scripts.util as util

def add_containment_potential(msm_savedir, temp_cv_r0, cv_grid, dcv, Ucv, dUcv, n_thresh):

    n = np.load(msm_savedir + "/psi1_n.npy").astype(float)
    Pn = n/np.sum(n)

    n_grid = scipy.interpolate.interp1d(temp_cv_r0, n)(cv_grid)

    # linearly extend potential at edges of domain to keep system where
    # potential is well-defined
    cv_grid_ext = []
    Ucv_ext = []
    for i in range(1, 10):
        if Ucv[i] < Ucv[i - 1]:
            # linearly extend potential at endpoints
            #m_L = (Ucv[i] - Ucv[i - 1])/(cv_grid[i] - cv_grid[i - 1])
            x0_L = cv_grid[i - 1]
            y0_L = Ucv[i - 1]
            
            m_L = np.sum(n_grid[i - 1:i - 1 + 4]*dUcv[i - 1:i - 1 + 4])/np.sum(n_grid[i - 1:i - 1 + 4])

            left_idx = i

            cv_ext_L = np.arange(x0_L - 15*dcv, x0_L, dcv)
            Ucv_ext_L = m_L*(cv_ext_L - x0_L) + y0_L
            cv_grid_ext.append(cv_ext_L)
            Ucv_ext.append(Ucv_ext_L)
            break

    for i in range(1, 10):
        if Ucv[len(Ucv) - i] > Ucv[len(Ucv) - i - 1]:
            # linearly extend potential at endpoints
            #m_R = (Ucv[len(Ucv) - i] - Ucv[len(Ucv) - i - 1])/(cv_grid[len(Ucv) - i] - cv_grid[len(Ucv) - i - 1])
            x0_R = cv_grid[len(Ucv) - i]
            y0_R = Ucv[len(Ucv) - i]

            m_R = np.sum(n_grid[len(Ucv) - i - 4:len(Ucv) - i]*dUcv[len(Ucv) - i - 4:len(Ucv) - i])/np.sum(n_grid[len(Ucv) - i - 4:len(Ucv) - i])

            right_idx = len(Ucv) - i

            cv_ext_R = np.arange(x0_R + dcv, x0_R + 15*dcv, dcv)
            Ucv_ext_R = m_R*(cv_ext_R - x0_R) + y0_R
            cv_grid_ext.append(cv_ext_R)
            Ucv_ext.append(Ucv_ext_R)
            break

    cv_grid_ext = np.concatenate([cv_ext_L, cv_grid[left_idx:right_idx + 1], cv_ext_R])
    Ucv_ext = np.concatenate([Ucv_ext_L, Ucv[left_idx:right_idx + 1], Ucv_ext_R])
    np.save(cg_savedir + "/Ucv_table.npy", np.array([cv_grid_ext, Ucv_ext]).T)

    plt.figure()
    plt.plot(cv_grid, Ucv, label="original")
    plt.plot(cv_grid_ext, Ucv_ext, 'k--', label="extended")
    plt.xlabel("TIC1")
    plt.ylabel(r"$U(\psi_1)$ potential")
    plt.legend()
    plt.savefig("plots/Ucv_ext.pdf")
    plt.savefig("plots/Ucv_ext.png")

    return cv_grid_ext, Ucv_ext

def get_Ucv_force(n_beads, Ucv_ext, cv_grid_ext, cv_coeff, cv_mean, feat_types, feat_idxs):


    feature_funcs = {"dist":"distance({}, {})", "invdist":"(1/distance({}, {}))", 
            "angle":"angle({}, {}, {})", "dih":"dihedral({}, {}, {}, {})"}

    cv_expr = "Table(Q); Q = "
    feat_idx = 0
    for i in range(len(feat_types)):
        feat_fun = feature_funcs[feat_types[i]]
        for j in range(cv_coeff.shape[0]):
            idxs = feat_idxs[i][j]
            idxs += 1

            feat_coeff = cv_coeff[feat_idx,0]
            feat_mean = cv_mean[feat_idx]

            #if feat_coeff < 0:
            #    sign1 = "-"
            #else:
            #    sign1 = "+"

            #if feat_mean < 0:
            #    sign2 = "-"
            #else:
            #    sign2 = "+"

            feat_explicit = feat_fun.format(*idxs)
            cv_expr += "c{}*({} - b{}) ".format(feat_idx + 1, feat_explicit, feat_idx + 1)
                
            #if i == 0:
            #    cv_expr += "{:.5f}*(distance(p{}, p{}) {} {:.5f}) ".format(b_i, idx1, idx2, sign2, abs(mean_b_i))
            #elif i == cv_coeff.shape[0] - 1:
            #    cv_expr += "{} {:.5f}*(distance(p{}, p{}) {} {:.5f});".format(sign1, abs(b_i), idx1, idx2, sign2, abs(mean_b_i))
            #else:
            #    cv_expr += "{} {:.5f}*(distance(p{}, p{}) {} {:.5f}) ".format(sign1, abs(b_i), idx1, idx2, sign2, abs(mean_b_i))

            feat_idx += 1 

    Ucv_force = omm.CustomCompoundBondForce(n_beads, cv_expr)
    params = []
    for i in range(cv_coeff.shape[0]):
        Ucv_force.addPerBondParameter("c" + str(i + 1))
        Ucv_force.addPerBondParameter("b" + str(i + 1))

        params.append(cv_coeff[i,0])
        params.append(cv_mean[i])

    Ucv_force.addBond(np.arange(n_beads), params)
    Ucv_force.addFunction(Ucv_ext, cv_grid_ext[0], cv_grid_ext[-1])

    return Ucv_force

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs coarse-grain model of polymer')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('n_beads', type=int, help='Number of beads in polymer.')
    parser.add_argument('msm_savedir', type=str, help='MSM save directory.')
    parser.add_argument('n_cv_basis_funcs', type=int, help='Number of basis functions.')
    parser.add_argument('n_cv_test_funcs', type=int, help='Number of test functions.')
    parser.add_argument('--n_eigenvectors', type=int, default=1, help='Number of eigenfunctions used.')
    parser.add_argument('--fixed_bonded_terms', action="store_true", help='Fixed boned terms.')
    parser.add_argument('--starting_pdb', type=str, default="", help='Starting pdb filename.')
    parser.add_argument('--timestep', type=float, default=0.002, help='Simulation timestep (ps).')
    parser.add_argument('--nsteps_out', type=int, default=100, help='Number of steps between saves.')
    parser.add_argument('--run_idx', type=int, default=0, help='Run index.')
    parser.add_argument('--T', type=float, default=300, help='Temperature.')
    parser.add_argument('--n_steps', type=int, default=int(5e6), help='Number of steps.')
    parser.add_argument('--nocuda', action="store_true", default=False, help='Dont specify cuda.')
    args = parser.parse_args()


    #python run_cg_model.py c25 25 msm_dists 100 100 --run_idx 1 --T 300.00 --n_steps 1000

    #run /home/ajk8/code/implicit_force_field/polymer_scripts/run_cg_model c25 25 msm_dists 100 200 --fixed_bonded_terms --run_idx 1 --T 300.00 --n_steps 1000

    name = args.name
    n_beads = args.n_beads
    msm_savedir = args.msm_savedir
    n_cv_basis_funcs = args.n_cv_basis_funcs
    n_cv_test_funcs = args.n_cv_test_funcs
    M = args.n_eigenvectors
    fixed_bonded_terms = args.fixed_bonded_terms

    timestep = args.timestep*unit.picosecond
    #collision_rate = args.collision_rate/unit.picosecond
    nsteps_out = args.nsteps_out
    run_idx = args.run_idx
    T = args.T
    n_steps = args.n_steps
    #save_forces = args.save_forces
    #save_velocities = args.save_velocities
    cuda = not args.nocuda
    temperature = T*unit.kelvin
    beta = 1/(0.0083145*T)

    using_cv = True
    using_cv_r0 = False
    using_D2 = False

    cwd = os.getcwd()


    Ucg, cg_savedir, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(
            msm_savedir, n_beads, M, beta, fixed_bonded_terms,
            using_cv, using_cv_r0, using_D2, n_cv_basis_funcs, n_cv_test_funcs)

    cg_savedir += "_crossval_5"

    # create tabulated function of collective variable 
    temp_cv_r0 = np.load(msm_savedir + "/psi1_mid_bin.npy")
    psi1_min, psi1_max = temp_cv_r0.min(), temp_cv_r0.max()

    cv_grid = np.linspace(psi1_min, psi1_max, 200)
    dcv = np.abs(cv_grid[1] - cv_grid[0])

    cv_coeff = np.load(msm_savedir + "/tica_eigenvects.npy")[:,:M]
    cv_mean = np.load(msm_savedir + "/tica_mean.npy")

    pair_idxs = []
    for i in range(n_beads - 1):
        for j in range(i + 4, n_beads):
            pair_idxs.append([i, j])
    pair_idxs = np.array(pair_idxs)

    coeff = np.load(cg_savedir + "/rdg_cstar.npy")
    D = 1/coeff[-1] # TODO turn into collision rate. Does mass matter?

    Ucv = np.zeros(len(cv_grid))
    for i in range(len(coeff) - 1):
        Ucv += coeff[i]*Ucg.cv_U_funcs[i](cv_grid)
    Ucv -= Ucv.min()
    dUcv = np.diff(Ucv)/dcv

    if os.path.exists(cg_savedir + "/Ucv_table.npy"):
        data = np.load(cg_savedir + "/Ucv_table.npy")
        cv_grid_ext = data[:,0]
        Ucv_ext = data[:,1]
    else:
        n_thresh = 500 
        cv_grid_ext, Ucv_ext = add_containment_potential(msm_savedir, temp_cv_r0, cv_grid, dcv, Ucv, dUcv, n_thresh)

    Ucv_force = get_Ucv_force(n_beads, Ucv_ext, cv_grid_ext, cv_coeff, cv_mean, pair_idxs)


    ###################################################
    # Run production 
    ###################################################
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    os.chdir(rundir)

    starttime = time.time()
    ref_pdb = app.PDBFile(ini_pdb_file)
    topology, positions = ref_pdb.topology, ref_pdb.positions

    templates = sop.util.template_dict(topology, n_beads)

    system = forcefield.createSystem(topology, ignoreExternalBonds=True, residueTemplates=templates)

    sop.run.production(topology, positions, ensemble, temperature, timestep,
            collision_rate, pressure, n_steps, nsteps_out, ff_files,
            min_name, log_name, traj_name, final_state_name, cutoff, templates,
            prev_state_name=prev_state_name,
            nonbondedMethod=nonbondedMethod, minimize=minimize, cuda=cuda,
            more_reporters=more_reporters, use_switch=False, r_switch=None,
            forcefield=forcefield)
