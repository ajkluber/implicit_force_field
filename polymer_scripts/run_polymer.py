import os
import shutil
import time
import argparse
import numpy as np

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import mdtraj as md

import simulation.openmm as sop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('n_beads', type=int, help='Number of beads in polymer.')
    parser.add_argument('ply_potential', type=str, help='Interactions for polymer.')
    parser.add_argument('slv_potential', type=str, help='Interactions for solvent.')
    parser.add_argument('--eps_ply', type=float, default=0, help='Polymer parameters.')
    parser.add_argument('--eps_slv', type=float, default=0, help='Solvent parameters.')
    parser.add_argument('--target_volume', type=float, default=-1, help='Target volume (nm^3).')
    parser.add_argument('--run_idx', type=int, default=0, help='Run index.')
    parser.add_argument('--T', type=float, default=300, help='Temperature.')
    parser.add_argument('--n_steps', type=int, default=int(5e6), help='Number of steps.')
    parser.add_argument('--save_forces', action="store_true", help='Save forces.')
    parser.add_argument('--save_velocities', action="store_true", help='Save velocities.')
    parser.add_argument('--nocuda', action="store_true", default=False, help='Dont specify cuda.')
    args = parser.parse_args()

    #python run_polymer.py c25 25 LJ LJ --eps_ply 1 --eps_slv 1 --run_idx 1 --T 300.00 --n_steps 1000

    name = args.name
    n_beads = args.n_beads
    ply_potential = args.ply_potential
    slv_potential = args.slv_potential
    eps_ply_mag = args.eps_ply
    eps_slv_mag = args.eps_slv
    target_volume = args.target_volume
    run_idx = args.run_idx
    T = args.T
    n_steps = args.n_steps
    save_forces = args.save_forces
    save_velocities = args.save_velocities
    cuda = not args.nocuda
    
    assert ply_potential in ["LJ", "wca"]
    assert slv_potential in ["LJ", "CS"]

    cwd = os.getcwd()

    refT = 300.
    ensemble = "NVT"
    cutoff = 0.9*unit.nanometers
    vdwCutoff = 0.9*unit.nanometers
    r_switch = 0.7*unit.nanometers

    ### Interaction parameters 
    #ply_potential = "LJ"
    #slv_potential = "LJ"

    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    if ply_potential == "LJ":
        if eps_ply_mag == 0:
            raise IOError("--eps_ply must be specified for LJ polymer potential")
        eps_ply = eps_ply_mag*unit.kilojoule_per_mole

    if slv_potential == "LJ":
        if eps_slv_mag == 0:
            raise IOError("--eps_slv must be specified for LJ solvent potential")
        eps_slv, sigma_slv, mass_slv = sop.util.LJslv_params(eps_slv_mag)
    else:
        eps_slv, sigma_slv, B, r0, Delta, mass_slv = sop.build_ff.CS_water_params()

    ff_filename = "ff_c{}.xml".format(n_beads)
    sop.util.add_elements(mass_slv, mass_ply)

    # directory structure
    Hdir = sop.util.get_Hdir(name, ply_potential, slv_potential, eps_ply_mag, eps_slv_mag)
    Pdir = Hdir + "/pressure_equil"
    Tdir = Hdir + "/T_{:.2f}".format(T)
    Vdir = Tdir + "/volume_equil"
    rundir_str = lambda idx: Tdir + "/run_{}".format(idx)


    ### Determine if trajectories exist for this run
    all_trajfiles_exist = lambda idx1, idx2: np.all([os.path.exists(rundir_str(idx1) + "/" + x) for x in sop.util.output_filenames(name, idx2)])

    # if run idx is not specified create new run directory 
    if run_idx == 0:
        run_idx = 1
        while all_trajfiles_exist(run_idx, 1):
            run_idx += 1

    rundir = rundir_str(run_idx)
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    if not os.path.exists(Pdir):
        os.mkdir(Pdir)
    if not os.path.exists(Vdir):
        os.mkdir(Vdir)
    if not os.path.exists(Tdir):
        os.mkdir(Tdir)

    # If trajectories exist in run directory, extend the last one.
    traj_idx = 1
    while all_trajfiles_exist(run_idx, traj_idx):
        traj_idx += 1

    if traj_idx == 1:
        minimize = True
    else:
        minimize = False

    # The pressure and unitcell dimensions must reproduce the density of 
    # the reference state point.     

    # When we coarse-grain, we change the forces in the system. Therefore we have to
    # find the pressure that reproduces the density (average box dimensions) of 
    # a reference system (water at 300K). Then simulations at different

    # We need to determine. Each cg Hamiltonian requires 
    # Then we can equilsimulation at different temperatures can 
    #   - sure the density is correct at this temperature (for this Hamiltonian).

    ###################################################
    # Determine reasonable pressure for model
    ###################################################
    os.chdir(Hdir)
    peq_state_name = Pdir + "/" + "final_state.xml"
    peq_log = Pdir + "/pressure_in_atm_vs_step.npy"
    if not (os.path.exists(peq_log) and os.path.exists(peq_state_name)):
        print "Reference pressure search"
        os.chdir(Pdir)
        shutil.copy(cwd + "/" + name + "_min.pdb", name + "_min.pdb")

        ref_pdb = md.load(name + "_min.pdb")
        if target_volume == -1:
            # reference structure has density of water
            target_volume = ref_pdb.unitcell_volumes[0]

        sop.build_ff.polymer_in_solvent(n_beads, ply_potential, slv_potential,
                saveas=ff_filename, eps_ply=eps_ply, sigma_ply=sigma_ply, mass_ply=mass_ply,
                eps_slv=eps_slv, sigma_slv=sigma_slv, mass_slv=mass_slv)

        # adaptive change pressure in order to get target unitcell volume (density). 
        print "  running adaptive simulations..."
        sop.run.adaptively_find_best_pressure(target_volume, ff_filename, name,
                n_beads, cutoff, r_switch, refT, save_forces=save_forces,
                cuda=cuda)
        os.chdir("..")
    else:
        print "Loading reference pressure"
    pressure = np.load(Pdir + "/pressure_in_atm_vs_step.npy")[-1]*unit.atmosphere
    os.chdir(cwd)

    ###################################################
    # Equilibrate unitecell volume at T and P
    ###################################################
    os.chdir(Tdir)
    veq_state_name = Vdir + "/final_state_nvt.xml"
    if not os.path.exists(veq_state_name):
        print "Unitcell volume equilibration"
        os.chdir("volume_equil")

        # let volume equilibrate at this pressure
        shutil.copy(cwd + "/" + name + "_min.pdb", name + "_min.pdb")
        sop.build_ff.polymer_in_solvent(n_beads, ply_potential, slv_potential,
                saveas=ff_filename, eps_ply=eps_ply, sigma_ply=sigma_ply,
                mass_ply=mass_ply, eps_slv=eps_slv, sigma_slv=sigma_slv,
                mass_slv=mass_slv)

        print "  equilibrating V at this T and P..."
        sop.run.equilibrate_unitcell_volume(pressure, ff_filename, name,
                n_beads, refT, T, cutoff, r_switch, peq_state_name, cuda=cuda)

        os.chdir("..")

    if traj_idx == 1:
        if ensemble == "NVT":
            prev_state_name = Vdir + "/final_state_nvt.xml"
        else:
            prev_state_name = Vdir + "/final_state_npt.xml"
    else:
        prev_state_name = name + "_final_state_{}.xml".format(traj_idx - 1)
    os.chdir(cwd)

    ###################################################
    # Run production 
    ###################################################
    os.chdir(rundir)

    if traj_idx == 1:
        shutil.copy(Vdir + "/" + name + "_min.pdb", name + "_min.pdb")

    # save force field
    sop.build_ff.polymer_in_solvent(n_beads, ply_potential, slv_potential,
            saveas=ff_filename, eps_ply=eps_ply, sigma_ply=sigma_ply, mass_ply=mass_ply,
            eps_slv=eps_slv, sigma_slv=sigma_slv, mass_slv=mass_slv)

    min_name, log_name, traj_name, final_state_name = sop.util.output_filenames(name, traj_idx)

    # simulation parameters
    nsteps_out = 100
    temperature = T*unit.kelvin
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond

    starttime = time.time()
    topology, positions = sop.util.get_starting_coordinates(name, traj_idx)
    templates = sop.util.template_dict(topology, n_beads)

    # reporters for forces and velocities
    more_reporters = []
    if save_forces:
        more_reporters.append(sop.additional_reporters.ForceReporter(name + "_forces_{}.dat".format(traj_idx), nsteps_out))
    if save_velocities:
        more_reporters.append(sop.additional_reporters.VelocityReporter(name + "_vels_{}.dat".format(traj_idx), nsteps_out))

    # Run simulation
    print "Running production..."
    sop.run.production(topology, positions, ensemble, temperature, timestep,
            collision_rate, pressure, n_steps, nsteps_out, ff_filename,
            min_name, log_name, traj_name, final_state_name, cutoff, templates,
            prev_state_name=prev_state_name,
            nonbondedMethod=app.CutoffPeriodic, minimize=minimize, cuda=cuda,
            more_reporters=more_reporters, use_switch=True, r_switch=r_switch)

    stoptime = time.time()
    with open("running_time.log", "w") as fout:
        fout.write("{} steps took {} min".format(n_steps, (stoptime - starttime)/60.))
    print "{} steps took {} min".format(n_steps, (stoptime - starttime)/60.)
