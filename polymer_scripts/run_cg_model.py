from __future__ import print_function
import os
import glob
import sys
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

def add_containment_potential(msm_savedir, temp_cv_r0, cv_grid, dcv, Ucv, dUcv, n_thresh, savedir, psi_lims):

    psi1_min, psi1_max = psi_lims

    n = np.load(msm_savedir + "/psi1_n.npy").astype(float)
    Pn = n/np.sum(n)

    n_grid = scipy.interpolate.interp1d(temp_cv_r0, n, fill_value=0, bounds_error=False)(cv_grid)

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
    plt.axvline(psi1_min, ls='--', color='r')
    plt.axvline(psi1_max, ls='--', color='r')
    plt.xlabel("TIC1")
    plt.ylabel(r"$U(\psi_1)$ potential")
    plt.legend()
    plt.savefig(savedir + "/Ucv_ext.pdf")
    plt.savefig(savedir + "/Ucv_ext.png")

    return cv_grid_ext, Ucv_ext

def get_Ucv_force(n_beads, Ucv_ext, cv_grid_ext, cv_coeff, cv_mean, feat_types, feat_idxs):
    feature_funcs = {"dist":"distance(p{}, p{})", "invdist":"(1/distance(p{}, p{}))", 
            "angle":"angle(p{}, p{}, p{})", "dih":"dihedral(p{}, p{}, p{}, p{})"}


    cv_expr = "Table("
    pr_cv_expr = "Table("
    #cv_expr = ""
    #pr_cv_expr = ""
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
                cv_expr += " + {:.5f})".format(abs(feat_mean))
                pr_cv_expr += " + {:.5f})".format(abs(feat_mean))
            else:
                cv_expr += " - {:.5f})".format(abs(feat_mean))
                pr_cv_expr += " - {:.5f})".format(abs(feat_mean))

            feat_idx += 1 

    cv_expr += ");"
    pr_cv_expr += ");"
    #cv_expr += ";"
    #pr_cv_expr += ";"

    Ucv_force = omm.CustomCompoundBondForce(n_beads, cv_expr)
    Ucv_force.addBond(atm_to_p_idx)
    #Ucv_force.addFunction("Table", Ucv_ext, cv_grid_ext[0], cv_grid_ext[-1])
    Table_func = omm.Continuous1DFunction(Ucv_ext, cv_grid_ext[0], cv_grid_ext[-1])
    Ucv_force.addTabulatedFunction("Table", Table_func)

    return Ucv_force, pr_cv_expr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs coarse-grain model of polymer')
    parser.add_argument('msm_savedir', type=str, help='MSM save directory.')
    parser.add_argument("cg_method", type=str, help='Coarse-graining method.')
    parser.add_argument("--psi_dims", type=int, default=1)
    parser.add_argument("--a_coeff", type=float, default=None, help='Diffusion coefficient used in eigenpair.')
    parser.add_argument("--using_cv", action="store_true")
    parser.add_argument("--n_basis", type=int, default=-1)
    parser.add_argument("--n_test", type=int, default=-1)
    parser.add_argument("--n_pair_gauss", type=int, default=-1)
    parser.add_argument("--pair_symmetry", type=str, default=None)
    parser.add_argument("--bond_cutoff", type=int, default=4)
    parser.add_argument("--lin_pot", action="store_true")
    parser.add_argument("--fix_back", action="store_true")
    parser.add_argument("--fix_exvol", action="store_true")
    parser.add_argument('--coeff_file', type=str, default="rdg_fixed_sigma_cstar.npy", help='Specify file with coefficients.')

    parser.add_argument('--starting_pdb', type=str, default="", help='Starting pdb filename.')
    parser.add_argument('--walltime', type=float, default=7.8, help='Number hours to run.')
    parser.add_argument('--timestep', type=float, default=0.002, help='Simulation timestep (ps).')
    parser.add_argument('--collision_rate', type=float, default=0, help='Simulation collision_rate (1/ps).')
    parser.add_argument('--nsteps_out', type=int, default=100, help='Number of steps between saves.')
    parser.add_argument('--run_idx', type=int, default=0, help='Run index.')
    parser.add_argument('--T', type=float, default=300, help='Temperature.')
    parser.add_argument('--n_steps', type=int, default=int(5e6), help='Number of steps.')
    parser.add_argument('--platform', type=str, default=None, help='Specify platform (CUDA, CPU).')
    parser.add_argument('--dry_run', action="store_true", help='Dry run. No simulation.')
    args = parser.parse_args()
    
    n_beads = 25
    name = "c" + str(n_beads)
    #name = args.name
    #n_beads = args.n_beads

    msm_savedir = args.msm_savedir
    cg_method = args.cg_method
    M = args.psi_dims
    using_cv = args.using_cv
    n_cv_basis_funcs = args.n_basis
    n_cv_test_funcs = args.n_test
    n_pair_gauss = args.n_pair_gauss
    pair_symmetry = args.pair_symmetry
    bond_cutoff = args.bond_cutoff
    fix_back = args.fix_back
    fix_exvol = args.fix_exvol
    lin_pot = args.lin_pot

    using_U0 = fix_back or fix_exvol

    print(" ".join(sys.argv))

    timestep = args.timestep*unit.picosecond
    walltime = args.walltime*unit.hour
    collision_rate = args.collision_rate
    nsteps_out = args.nsteps_out
    run_idx = args.run_idx
    T = args.T
    a_coeff = args.a_coeff
    n_steps = args.n_steps
    platform = args.platform
    dry_run = args.dry_run
    temperature = T*unit.kelvin

    kb = 0.0083145
    beta = 1./(kb*T)

    if a_coeff is None:
        fixed_a = False
    else:
        fixed_a = True

    using_D2 = False
    n_cross_val_sets = 5

    if (n_cv_basis_funcs != -1) and (n_cv_test_funcs != -1):
        print("Since n_test ({}) and n_basis ({}) are specified -> using_cv=True".format(n_cv_test_funcs, n_cv_basis_funcs))
        using_cv = True
    else:
        if using_cv:
            raise ValueError("Please specify n_test and n_basis")

    if n_pair_gauss != -1:
        if not pair_symmetry in ["shared", "seq_sep", "unique"]:
            raise ValueError("Must specificy pair_symmetry")

    cg_savedir = util.test_Ucg_dirname(cg_method, M, using_U0, fix_back, fix_exvol,
            bond_cutoff, using_cv, n_cv_basis_funcs=n_cv_basis_funcs,
            n_cv_test_funcs=n_cv_test_funcs, a_coeff=a_coeff,
            n_pair_gauss=n_pair_gauss, cv_lin_pot=lin_pot,
            pair_symmetry=pair_symmetry)

    print(cg_savedir)
    #print(str(os.path.exists(cg_savedir + "/" + args.coeff_file)))
    #raise SystemExit

    Ucg, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg( msm_savedir,
            n_beads, M, beta, fix_back, fix_exvol, using_cv, using_D2,
            n_cv_basis_funcs, n_cv_test_funcs, n_pair_gauss, bond_cutoff,
            cv_lin_pot=lin_pot, a_coeff=a_coeff, pair_symmetry=pair_symmetry)

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

    if not os.path.exists(rundir):
        os.makedirs(rundir)

    # If trajectories exist in run directory, extend the last one.
    traj_idx = 1
    while all_trajfiles_exist(run_idx, traj_idx):
        traj_idx += 1

    if traj_idx == 1:
        minimize = True
    else:
        minimize = False

    min_name, log_name, traj_name, final_state_name = sop.util.output_filenames(name, traj_idx)

    coeff = np.load(cg_savedir + "/" + args.coeff_file)
    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()

    if using_cv:
        ####################################################
        # create collective variable potential
        ####################################################
        #coeff = np.load(cg_savedir + "/rdg_cstar.npy")
        #coeff = np.load(cg_savedir + "/rdg_fixed_sigma_cstar.npy")


        cv_coeff = np.load(msm_savedir + "/tica_eigenvects.npy")[:,:M]
        cv_mean = np.load(msm_savedir + "/tica_mean.npy")

        temp_cv_r0 = np.load(msm_savedir + "/psi1_mid_bin.npy")
        #psi1_min, psi1_max = temp_cv_r0.min(), temp_cv_r0.max()
        #cv_grid = np.linspace(psi1_min, psi1_max, 200)

        # set domain of potential
        # real min and max from data
        psinames = glob.glob(msm_savedir + "/run_*TIC_1.npy")
        psi1_min = np.min([ np.load(x).min() for x in psinames  ])
        psi1_max = np.max([ np.load(x).max() for x in psinames  ])
        cv_grid = np.linspace(1.3*cv_r0_basis.min(), 1.2*cv_r0_basis.max(), 200)

        dcv = np.abs(cv_grid[1] - cv_grid[0])

        Ucv = Ucg.Ucv_values(coeff, cv_grid)
        dUcv = np.diff(Ucv)/dcv

        # IS THIS NEEDED?
        # create tabulated function of collective variable 
        #if os.path.exists(cg_savedir + "/Ucv_table.npy") and False:
        #    data = np.load(cg_savedir + "/Ucv_table.npy")
        #    cv_grid_ext = data[:,0]
        #    Ucv_ext = data[:,1]
        #else:
        #    n_thresh = 500 
        #    cv_grid_ext, Ucv_ext = add_containment_potential(msm_savedir, temp_cv_r0, cv_grid, dcv, Ucv, dUcv, n_thresh, cg_savedir, (psi1_min, psi1_max))

        cv_grid_ext = cv_grid
        Ucv_ext = Ucv

        pair_idxs = []
        for i in range(n_beads - 1):
            for j in range(i + 4, n_beads):
                pair_idxs.append([i, j])
        pair_idxs = np.array(pair_idxs)

        feat_types = ["dist"]
        feat_idxs = [pair_idxs]

        Ucv_force, pr_cv_expr = get_Ucv_force(n_beads, Ucv_ext, cv_grid_ext, cv_coeff, cv_mean, feat_types, feat_idxs)
        #print(pr_cv_expr)

        plt.figure()
        plt.plot(cv_grid, Ucv)
        plt.xlabel(r"TIC1 $\psi_1$")
        plt.ylabel(r"$U_{\mathrm{cv}}(\psi_1)$ (kJ/mol)")
        plt.savefig(rundir + "/tab_Ucv_func.pdf")
        plt.savefig(rundir + "/tab_Ucv_func.png")
    else:
        if pair_symmetry == "shared":
            pass
        elif pair_symmetry == "shared":
            pass
        elif pair_symmetry == "shared":
            pass
        else:
            print("ERROR")

    ###################################################
    # Run production 
    ###################################################
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
    ff_kwargs["eps_ply"] = coeff[0]*0.5*unit.kilojoule_per_mole
    ff_kwargs["sigma_ply"] = sigma_ply

    ff_filename = "ff_cgs.xml"
    sop.build_ff.polymer_in_solvent(n_beads, "r12", "NONE",
            saveas=ff_filename, bond_cutoff=bond_cutoff - 1,
            **ff_kwargs)

    forcefield = app.ForceField(ff_filename)

    system = forcefield.createSystem(topology, ignoreExternalBonds=True, residueTemplates=templates)

    if using_cv:
        system.addForce(Ucv_force)
    else:
        # add pairwise potentials
        pass

    if collision_rate == 0:
        # gamma = m/D ? does temperature enter?
        collision_rate = ((mass_ply/unit.amu)/D)/unit.picosecond
    else:
        collision_rate = collision_rate/unit.picosecond

    ensemble = "NVT"
    firstframe_name = name + "_min_" + str(traj_idx) + ".pdb"

    #save_E_groups = [0, 1, 2, 3, 4]
    save_E_groups = []

    if traj_idx == 1:
        prev_state_name = None
    else:
        positions = None
        prev_state_name = name + "_final_state_" + str(traj_idx - 1) + ".xml"

    if not dry_run:
        print("Running production...")
        sys.stdout.flush()
        sop.run.production(system, topology, ensemble, temperature, timestep,
                collision_rate, n_steps, nsteps_out, firstframe_name, log_name,
                traj_name, final_state_name, ini_state_name=prev_state_name,
                ini_positions=positions, minimize=minimize,
                use_platform=platform, walltime=walltime,
                save_E_groups=save_E_groups)

        stoptime = time.time()
        with open("running_time.log", "w") as fout:
            fout.write("{} steps took {} min".format(n_steps, (stoptime - starttime)/60.))
        print("{} steps took {} min".format(n_steps, (stoptime - starttime)/60.))

    raise SystemExit

    #Esim = np.loadtxt("c25_1.log", usecols=(1,), delimiter=",")
    #Eterms = np.loadtxt("E_terms.dat")
    Eterms = np.load("E_terms.npy")
    Eex = Eterms[:,1]
    Eb = Eterms[:,2]
    Ea = Eterms[:,3]
    Ecv = Eterms[:,5]

    traj = md.load("c25_traj_1.dcd", top="c25_noslv_min.pdb")
    bond_r = md.compute_distances(traj, np.array([[i , i + 1] for i in range(24) ]))
    angle_theta = md.compute_angles(traj, np.array([[i , i + 1, i + 2] for i in range(23) ]))

    pair_idxs = []
    for i in range(n_beads - 1):
        for j in range(i + 4, n_beads):
            pair_idxs.append([i, j])
    pair_idxs = np.array(pair_idxs)
    rij = md.compute_distances(traj, pair_idxs)

    Uex_md = np.sum(0.5*((0.373/rij)**12), axis=1)
    Ub_md = np.sum(0.5*334720.0*((bond_r - 0.153)**2), axis=1)
    Ua_md = np.sum(0.5*462.0*((angle_theta - 1.938)**2), axis=1)

    xyz_traj = np.reshape(traj.xyz, (-1, 75))
    cv_traj = Ucg.calculate_cv(xyz_traj)

    U0 = Ucg.potential_U0(xyz_traj, cv_traj, sumterms=False)
    #Ub, Ua, Uex = U0
    Ub, Ua = U0

    U_k = Ucg.potential_U1(xyz_traj, cv_traj)
    Uex = coeff[0]*Uex_md
    Ucv_calc = np.einsum("k,tk->t", coeff[1:], U_k[:,1:])

    simE = [Eb, Ea, Eex, Ecv]
    calcE = [Ub, Ua, Uex, Ucv_calc]
    labels = ["Bond", "Angle", "Excl.", "CV"]

    #((ax1, ax2), (ax3, ax4))
    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    idx = 0 
    for i in range(2):
        for j in range(2):
            ax = axes[i,j]
            x = simE[idx]
            y = calcE[idx]
    
            minx = min([ x.min(), y.min() ])
            maxx = max([ x.max(), y.max() ])
            ax.plot([minx, maxx], [minx, maxx], 'k')
            ax.plot(x, y, 'b.')
            ax.set_xlim(minx, maxx)
            ax.set_ylim(minx, maxx)
            ax.set_title(labels[idx])

            if idx in [0, 2]:
                ax.set_ylabel("Calculated")

            if idx in [2, 3]:
                ax.set_xlabel("Simulation")

            idx += 1

    fig.savefig("compare_Eterms.pdf")
    fig.savefig("compare_Eterms.png")
