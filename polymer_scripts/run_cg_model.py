from __future__ import print_function
import os
import shutil
import time
import argparse
import numpy as np
import scipy.interpolate
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import mdtraj as md

import simulation.openmm as sop

import implicit_force_field.polymer_scripts.util as util

def add_containment_potential(msm_savedir, temp_cv_r0, cv_grid, dcv, Ucv, dUcv, n_thresh, savedir):

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
    plt.savefig(savedir + "/Ucv_ext.pdf")
    plt.savefig(savedir + "/Ucv_ext.png")

    return cv_grid_ext, Ucv_ext

def get_Ucv_force(n_beads, Ucv_ext, cv_grid_ext, cv_coeff, cv_mean, feat_types, feat_idxs):
    feature_funcs = {"dist":"distance(p{}, p{})", "invdist":"(1/distance(p{}, p{}))", 
            "angle":"angle(p{}, p{}, p{})", "dih":"dihedral(p{}, p{}, p{}, p{})"}


    cv_expr = "Table(Q); Q = "
    #cv_expr = "-100*("
    pr_cv_expr = ""
    feat_idx = 0
    atm_to_p_idx = []
    p_idx = -np.ones(n_beads, int)
    curr_p_idx = 1
    for i in range(len(feat_types)):
        feat_fun = feature_funcs[feat_types[i]]
        for j in range(cv_coeff.shape[0]):
            idxs = feat_idxs[i][j]

            expr_idxs = []
            for n in range(len(idxs)): 
                if p_idx[idxs[n]] == -1:
                    p_idx[idxs[n]] = curr_p_idx
                    atm_to_p_idx.append(int(idxs[n]))
                    curr_p_idx += 1
                expr_idxs.append(p_idx[idxs[n]])
            #print("{} -> {}      {} -> {}".format(idxs[0], expr_idxs[0], idxs[1], expr_idxs[1]))   # DEBUGGING

            feat_explicit = feat_fun.format(*expr_idxs)

            feat_coeff = cv_coeff[feat_idx,0]
            feat_mean = cv_mean[feat_idx]

            if feat_idx > 0:
                if feat_coeff < 0:
                    cv_expr += " - {:.5f}*(".format(abs(feat_coeff))
                    pr_cv_expr += "\n - {:.5f}*(".format(abs(feat_coeff))
                else:
                    cv_expr += " + {:.5f}*(".format(abs(feat_coeff))
                    pr_cv_expr += "\n + {:.5f}*(".format(abs(feat_coeff))
            else:
                if feat_coeff < 0:
                    cv_expr += "-{:.5f}*(".format(abs(feat_coeff))
                    pr_cv_expr += "-{:.5f}*(".format(abs(feat_coeff))
                else:
                    cv_expr += "{:.5f}*(".format(abs(feat_coeff))
                    pr_cv_expr += "{:.5f}*(".format(abs(feat_coeff))

            cv_expr += feat_explicit
            pr_cv_expr += feat_explicit

            if feat_mean < 0:
                cv_expr += " - {:.5f})".format(abs(feat_mean))
                pr_cv_expr += " - {:.5f})".format(abs(feat_mean))
            else:
                cv_expr += " + {:.5f})".format(abs(feat_mean))
                pr_cv_expr += " + {:.5f})".format(abs(feat_mean))

            feat_idx += 1 

    #cv_expr += ")"
    cv_expr += ";"
    pr_cv_expr += ";"

    Ucv_force = omm.CustomCompoundBondForce(n_beads, cv_expr)
    Ucv_force.addBond(atm_to_p_idx)
    Ucv_force.addFunction("Table", Ucv_ext, cv_grid_ext[0], cv_grid_ext[-1])

    return Ucv_force, pr_cv_expr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs coarse-grain model of polymer')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('n_beads', type=int, help='Number of beads in polymer.')
    parser.add_argument('msm_savedir', type=str, help='MSM save directory.')
    parser.add_argument('n_cv_basis_funcs', type=int, help='Number of basis functions.')
    parser.add_argument('n_cv_test_funcs', type=int, help='Number of test functions.')
    parser.add_argument('--n_eigenvectors', type=int, default=1, help='Number of eigenfunctions used.')
    parser.add_argument('--force_matching', action="store_true", help='Solution is from force matching.')
    parser.add_argument('--fixed_bonded_terms', action="store_true", help='Fixed boned terms.')
    parser.add_argument('--starting_pdb', type=str, default="", help='Starting pdb filename.')
    parser.add_argument('--timestep', type=float, default=0.002, help='Simulation timestep (ps).')
    parser.add_argument('--collision_rate', type=float, default=0, help='Simulation collision_rate (1/ps).')
    parser.add_argument('--nsteps_out', type=int, default=100, help='Number of steps between saves.')
    parser.add_argument('--run_idx', type=int, default=0, help='Run index.')
    parser.add_argument('--T', type=float, default=300, help='Temperature.')
    parser.add_argument('--n_steps', type=int, default=int(5e6), help='Number of steps.')
    parser.add_argument('--platform', type=str, default=None, help='Specify platform (CUDA, CPU).')
    parser.add_argument('--dry_run', action="store_true", help='Dry run. No simulation.')
    args = parser.parse_args()


    #python run_cg_model.py c25 25 msm_dists 100 100 --run_idx 1 --T 300.00 --n_steps 1000

    #run /home/ajk8/code/implicit_force_field/polymer_scripts/run_cg_model c25 25 msm_dists 100 200 --fixed_bonded_terms --T 300.00 --n_steps 1000 --collision_rate 1 --platform CPU --run_idx 1

    #python /home/ajk8/code/implicit_force_field/polymer_scripts/run_cg_model c25 25 msm_dists 100 200 --fixed_bonded_terms --T 300.00 --n_steps 1000 --collision_rate 1 --platform CPU --run_idx 1

    #python /home/ajk8/code/implicit_force_field/polymer_scripts/run_cg_model c25 25 msm_dists 40 100 --fixed_bonded_terms --force_matching --T 300.00 --n_steps 1000 --collision_rate 1 --platform CPU --run_idx 1


    name = args.name
    n_beads = args.n_beads
    msm_savedir = args.msm_savedir
    n_cv_basis_funcs = args.n_cv_basis_funcs
    n_cv_test_funcs = args.n_cv_test_funcs
    M = args.n_eigenvectors
    from_fm = args.force_matching
    fixed_bonded_terms = args.fixed_bonded_terms

    timestep = args.timestep*unit.picosecond
    collision_rate = args.collision_rate
    nsteps_out = args.nsteps_out
    run_idx = args.run_idx
    T = args.T
    n_steps = args.n_steps
    #save_forces = args.save_forces
    #save_velocities = args.save_velocities
    #cuda = not args.nocuda
    use_platform = args.platform
    dry_run = args.dry_run
    temperature = T*unit.kelvin
    beta = 1/(0.0083145*T)

    using_cv = True
    using_cv_r0 = False
    using_D2 = False

    if from_fm:
        dir_stem = "Ucg_FM"
    else:
        dir_stem = "Ucg_eigenpair"

    Ucg, cg_savedir, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(
            msm_savedir, n_beads, M, beta, fixed_bonded_terms,
            using_cv, using_cv_r0, using_D2, n_cv_basis_funcs, n_cv_test_funcs,
            cg_savedir=dir_stem)

    cg_savedir += "_crossval_5"

    cwd = os.getcwd()
    Hdir = cwd + "/" + cg_savedir

    rundir_str = lambda idx: Hdir + "/run_{}".format(idx)

    # Determine if trajectories exist for this run
    all_trajfiles_exist = lambda idx1, idx2: np.all([os.path.exists(rundir_str(idx1) + "/" + x) for x in sop.util.output_filenames(name, idx2)])

    # if run idx is not specified create new run directory 
    if run_idx == 0:
        run_idx = 1
        while all_trajfiles_exist(run_idx, 1):
            run_idx += 1

    rundir = rundir_str(run_idx)

    # If trajectories exist in run directory, extend the last one.
    traj_idx = 1
    #while all_trajfiles_exist(run_idx, traj_idx):
    #    traj_idx += 1

    if traj_idx == 1:
        minimize = True
    else:
        minimize = False

    min_name, log_name, traj_name, final_state_name = sop.util.output_filenames(name, traj_idx)

    ####################################################
    # create potential
    ####################################################
    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()

    # create tabulated function of collective variable 
    temp_cv_r0 = np.load(msm_savedir + "/psi1_mid_bin.npy")
    psi1_min, psi1_max = temp_cv_r0.min(), temp_cv_r0.max()

    cv_grid = np.linspace(psi1_min, psi1_max, 200)
    dcv = np.abs(cv_grid[1] - cv_grid[0])

    cv_coeff = np.load(msm_savedir + "/tica_eigenvects.npy")[:,:M]
    cv_mean = np.load(msm_savedir + "/tica_mean.npy")


    coeff = np.load(cg_savedir + "/rdg_cstar.npy")

    if not from_fm and not Ucg.fixed_a_coeff:
        D = 1/coeff[-1] 

        Ucv = np.zeros(len(cv_grid))
        for i in range(len(coeff) - 1):
            Ucv += coeff[i]*Ucg.cv_U_funcs[i](cv_grid)
        Ucv -= Ucv.min()
    else:
        Ucv = np.zeros(len(cv_grid))
        for i in range(len(coeff)):
            Ucv += coeff[i]*Ucg.cv_U_funcs[i](cv_grid)
        Ucv -= Ucv.min()

    dUcv = np.diff(Ucv)/dcv

    if os.path.exists(cg_savedir + "/Ucv_table.npy"):
        data = np.load(cg_savedir + "/Ucv_table.npy")
        cv_grid_ext = data[:,0]
        Ucv_ext = data[:,1]
    else:
        n_thresh = 500 
        cv_grid_ext, Ucv_ext = add_containment_potential(msm_savedir, temp_cv_r0, cv_grid, dcv, Ucv, dUcv, n_thresh, cg_savedir)

    pair_idxs = []
    for i in range(n_beads - 1):
        for j in range(i + 4, n_beads):
            pair_idxs.append([i, j])
    pair_idxs = np.array(pair_idxs)

    feat_types = ["dist"]
    feat_idxs = [pair_idxs]

    Ucv_force, pr_cv_expr = get_Ucv_force(n_beads, Ucv_ext, cv_grid_ext, cv_coeff, cv_mean, feat_types, feat_idxs)

    ###################################################
    # Run production 
    ###################################################
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    os.chdir(rundir)

    ini_pdb_file = name + "_noslv_min.pdb"
    if not os.path.exists(ini_pdb_file):
        shutil.copy("../../" + ini_pdb_file, ini_pdb_file)

    starttime = time.time()
    ref_pdb = app.PDBFile(ini_pdb_file)
    topology, positions = ref_pdb.topology, ref_pdb.positions

    templates = sop.util.template_dict(topology, n_beads)

    ff_kwargs = {}
    ff_kwargs["mass_ply"] = mass_ply
    ff_kwargs["sigma_ply"] = sigma_ply
    ff_kwargs["eps_ply"] = 0.5*unit.kilojoule_per_mole

    ff_filename = "ff_cgs.xml"
    sop.build_ff.polymer_in_solvent(n_beads, "r12", "NONE",
            saveas=ff_filename, **ff_kwargs)

    forcefield = app.ForceField(ff_filename)

    system = forcefield.createSystem(topology, ignoreExternalBonds=True, residueTemplates=templates)
    system.addForce(Ucv_force)

    if collision_rate == 0:
        # gamma = m/D ? does temperature enter?
        collision_rate = ((mass_ply/unit.amu)/D)/unit.picosecond
    else:
        collision_rate = collision_rate/unit.picosecond

    ensemble = "NVT"
    firstframe_name = name + "_min_" + str(traj_idx) + ".pdb"

    save_E_groups = [0, 1, 2, 3, 4]

    if not dry_run:
        print("Running production...")
        sop.run.production(system, topology, ensemble, temperature, timestep,
                collision_rate, n_steps, nsteps_out, firstframe_name, log_name,
                traj_name, final_state_name, ini_positions=positions,
                minimize=minimize, use_platform="CPU", save_E_groups=save_E_groups)

        stoptime = time.time()
        with open("running_time.log", "w") as fout:
            fout.write("{} steps took {} min".format(n_steps, (stoptime - starttime)/60.))
        print("{} steps took {} min".format(n_steps, (stoptime - starttime)/60.))

    #Esim = np.loadtxt("c25_1.log", usecols=(1,), delimiter=",")
    #Eterms = np.loadtxt("E_terms.dat")
    Eterms = np.load("E_terms.npy")
    Eex = Eterms[:,1]
    Eb = Eterms[:,2]
    Ea = Eterms[:,3]
    Ecv = Eterms[:,4]

    traj = md.load("c25_traj_1.dcd", top="c25_noslv_min.pdb")
    bond_r = md.compute_distances(traj, np.array([[i , i + 1] for i in range(24) ]))
    angle_theta = md.compute_angles(traj, np.array([[i , i + 1, i + 2] for i in range(23) ]))

    pair_idxs = []
    for i in range(n_beads - 1):
        for j in range(i + 4, n_beads):
            pair_idxs.append([i, j])
    pair_idxs = np.array(pair_idxs)
    rij = md.compute_distances(traj, pair_idxs)

    Uex_md = np.sum(0.5*((0.373/rij)**(12)), axis=1)
    Ub_md = np.sum(0.5*334720.0*((bond_r - 0.153)**2), axis=1)
    Ua_md = np.sum(0.5*462.0*((angle_theta - 1.938)**2), axis=1)


    xyz_traj = np.reshape(traj.xyz, (-1, 75))
    cv_traj = Ucg.calculate_cv(xyz_traj)

    U0 = Ucg.potential_U0(xyz_traj, cv_traj, sumterms=False)
    U1 = np.einsum("k,tk->t", coeff, Ucg.potential_U1(xyz_traj, cv_traj))


