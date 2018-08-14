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
import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('n_beads', type=int, help='Number of beads in polymer.')
    parser.add_argument('ply_potential', type=str, help='Interactions for polymer.')
    parser.add_argument('slv_potential', type=str, help='Interactions for solvent.')
    parser.add_argument('--eps_ply', type=float, default=0, help='Polymer parameters.')
    parser.add_argument('--eps_slv', type=float, default=0, help='Solvent parameters.')
    parser.add_argument('--run_idx', type=int, default=0, help='Run index.')
    parser.add_argument('--T', type=float, default=300, help='Temperature.')
    parser.add_argument('--n_steps', type=int, default=int(5e6), help='Number of steps.')
    parser.add_argument('--save_forces_and_vels', action="store_true", help='Save forces.')
    args = parser.parse_args()

    #python run_polymer.py c25 25 LJ LJ --eps_ply 1 --eps_slv 1 --run_idx 1 --T 300.00 --n_steps 1000

    name = args.name
    n_beads = args.n_beads
    ply_potential = args.ply_potential
    slv_potential = args.slv_potential
    eps_ply_mag = args.eps_ply
    eps_slv_mag = args.eps_slv
    run_idx = args.run_idx
    T = args.T
    n_steps = args.n_steps
    forces_and_velocities = args.save_forces_and_vels
    
    assert ply_potential in ["LJ", "wca"]
    assert slv_potential in ["LJ", "CS"]

    cwd = os.getcwd()

    refT = 300
    ensemble = "NVT"
    cutoff = 0.9*unit.nanometers
    vdwCutoff = 0.9*unit.nanometers
    r_switch = 0.7*unit.nanometers
    minimize = False

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
        eps_slv, sigma_slv, mass_slv = util.LJslv_params(eps_slv_mag)
    else:
        eps_slv, sigma_slv, B, r0, Delta, mass_slv = sop.build_ff.CS_water_params()

    ff_filename = "ff_c{}.xml".format(n_beads)
    util.add_elements(mass_slv, mass_ply)

    # directory structure
    Hdir = util.get_Hdir(name, ply_potential, slv_potential, eps_ply_mag, eps_slv_mag)
    #Hdir = cwd + "/{}_LJ_LJslv/eps_slv_{:.2f}".format(name, eps_slv_mag)
    Tdir = Hdir + "/T_{:.2f}".format(T)
    rundir_str = lambda idx: Tdir + "/run_{}".format(idx)

    print "Hdir:", Hdir 
    print "Tdir:", Tdir 
    print "rundir:", rundir_str

    ### Determine if trajectories exist for this run
    all_trajfiles_exist = lambda idx1, idx2: np.all([os.path.exists(rundir_str(idx1) + "/" + x) for x in util.output_filenames(name, idx2)])

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

    ### Get pressure
    # The pressure and unitcell dimensions must reproduce the density of 
    # the reference state point.     

    # When we coarse-grain, we change the forces in the system. Therefore we have to
    # find the pressure that reproduces the density (average box dimensions) of 
    # a reference system (water at 300K). Then simulations at different

    # We need to determine. Each cg Hamiltonian requires 
    # Then we can equilsimulation at different temperatures can 
    #   - sure the density is correct at this temperature (for this Hamiltonian).
    os.chdir(Hdir)

    Pdir = "pressure_equil"
    if not os.path.exists(Pdir + "/pressure.dat"):
        print "Reference pressure search"
        if not os.path.exists(Pdir):
            os.mkdir(Pdir)
        os.chdir(Pdir)
        shutil.copy(cwd + "/" + name + "_min.pdb", name + "_min.pdb")

        ref_pdb = md.load(name + "_min.pdb")
        target_volume = ref_pdb.unitcell_volumes[0]

        sop.build_ff.polymer_in_solvent(n_beads, ply_potential, slv_potential,
                saveas=ff_filename, eps_ply=eps_ply, sigma_ply=sigma_ply, mass_ply=mass_ply,
                eps_slv=eps_slv, sigma_slv=sigma_slv, mass_slv=mass_slv)

        # adaptive change pressure in order to get target unitcell volume (density). 
        print "  running adaptive simulations..."
        sop.run.adaptively_find_best_pressure(target_volume, ff_filename, name, n_beads, cutoff, r_switch, refT=refT)
        os.chdir("..")

    print "Loading reference pressure"
    #pressure =3931.122169*unit.atmosphere # found for c25_wca_CSslv
    pressure = np.loadtxt(Pdir + "/pressure.dat")[0]*unit.atmosphere
    #refT= float(np.loadtxt(P_str + "/temperature.dat"))

    os.chdir(cwd)

    ### Equilibrate unitcell volume at this temperature and pressure (if not the reference temperature).
    if not os.path.exists(Tdir):
        os.makedirs(Tdir)
    os.chdir(Tdir)

    equil_pdb_name = os.getcwd() + "/volume_equil/{}_fin_1.pdb".format(name)
    if not os.path.exists(equil_pdb_name):
        print "Unitcell volume equilibration"
        os.mkdir("volume_equil")
        os.chdir("volume_equil")
        shutil.copy(cwd + "/" + name + "_min.pdb", name + "_min.pdb")
        # let volume equilibrate at this pressure
        sop.build_ff.polymer_in_solvent(n_beads, ply_potential, slv_potential,
                saveas=ff_filename, eps_ply=eps_ply, sigma_ply=sigma_ply, mass_ply=mass_ply,
                eps_slv=eps_slv, sigma_slv=sigma_slv, mass_slv=mass_slv)

        print "  equilibrating volume at this pressure..."
        sop.run.equilibrate_unitcell_volume(pressure, ff_filename, name, n_beads, T, cutoff, r_switch)
        os.chdir("..")
    os.chdir(cwd)

    ### Run simulation
    os.chdir(rundir)

    if traj_idx == 1:
        shutil.copy(equil_pdb_name, name + "_min.pdb")

    # save force field
    #sop.build_ff.LJ_toy_polymer_LJ_water(n_beads, cutoff, solvent_params, saveas=ff_filename)
    sop.build_ff.polymer_in_solvent(n_beads, ply_potential, slv_potential,
            saveas=ff_filename, eps_ply=eps_ply, sigma_ply=sigma_ply, mass_ply=mass_ply,
            eps_slv=eps_slv, sigma_slv=sigma_slv, mass_slv=mass_slv)

    min_name, log_name, traj_name, lastframe_name = util.output_filenames(name, traj_idx)

    # simulation parameters
    nsteps_out = 100
    temperature = T*unit.kelvin
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond

    starttime = time.time()
    topology, positions = util.get_starting_coordinates(name, traj_idx)
    templates = util.template_dict(topology, n_beads)

    #forces_and_velocities = True
    more_reporters = []
    if forces_and_velocities:
        more_reporters.append(sop.additional_reporters.ForceReporter(name + "_forces_{}.dat".format(traj_idx), nsteps_out))
        more_reporters.append(sop.additional_reporters.VelocityReporter(name + "_vels_{}.dat".format(traj_idx), nsteps_out))

    # Run simulation
    print "Running production..."
    sop.run.production(topology, positions, ensemble, temperature, timestep,
            collision_rate, pressure, n_steps, nsteps_out, ff_filename,
            min_name, log_name, traj_name, lastframe_name, cutoff, templates,
            nonbondedMethod=app.CutoffPeriodic, minimize=minimize, cuda=True, 
            more_reporters=more_reporters, use_switch=True, r_switch=r_switch)

    stoptime = time.time()
    with open("running_time.log", "w") as fout:
        fout.write("{} steps took {} min".format(n_steps, (stoptime - starttime)/60.))
    print "{} steps took {} min".format(n_steps, (stoptime - starttime)/60.)
