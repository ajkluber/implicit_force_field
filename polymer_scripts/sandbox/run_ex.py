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
import implicit_force_field.polymer_scripts.util as util

if __name__ == "__main__":
    name = "c25"
    n_beads = 25
    eps_ply_mag = 0.1
    eps_slv_mag = 1
    run_idx = 1
    traj_idx = 1
    T = 300
    n_steps = 10000
    forces_and_velocities = False
    ply_potential = "LJ"
    slv_potential = "LJ"
    dynamics = "Langevin"
    ensemble = "NPT"
    
    cwd = os.getcwd()

    refT = 300
    ensemble = "NVT"
    cutoff = 0.9*unit.nanometers
    vdwCutoff = 0.9*unit.nanometers
    r_switch = 0.7*unit.nanometers
    minimize = False
    eps_slv, sigma_slv, mass_slv = util.LJslv_params(eps_slv_mag)

    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    eps_ply = eps_ply_mag*unit.kilojoule_per_mole

    ff_filename = "ff_c{}.xml".format(n_beads)
    util.add_elements(mass_slv, mass_ply)

    pressure = np.loadtxt("pressure.dat")[0]*unit.atmosphere

    nsteps_out = 100
    temperature = T*unit.kelvin
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond
    more_reporters = []

    pdb = app.PDBFile(name + "_min.pdb")
    topology = pdb.topology
    positions = pdb.positions
    templates = util.template_dict(topology, n_beads)

    #properties = {'DeviceIndex': '0'}
    platform = omm.Platform.getPlatformByName('CPU') 

    min_name = name + "_min_{}.pdb".format(traj_idx)
    log_name = name + "_{}.log".format(traj_idx)
    traj_name = name + "_traj_{}.dcd".format(traj_idx)
    lastframe_name = name + "_fin_{}.pdb".format(traj_idx)

    forcefield = app.ForceField(ff_filename)

    system = forcefield.createSystem(topology,
            nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff,
            ignoreExternalBonds=True, residueTemplates=templates)

    nb_force = system.getForce(0) # assume nonbonded interactions are first force
    nb_force.setUseSwitchingFunction(True)
    if r_switch == 0:
        raise IOError("Set switching distance")
    else:
        nb_force.setSwitchingDistance(r_switch/unit.nanometer)

    integrator = omm.LangevinIntegrator(temperature, collision_rate, timestep)

    system.addForce(omm.MonteCarloBarostat(pressure, temperature))

    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
    simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
        step=True, potentialEnergy=True, kineticEnergy=True, temperature=True,
        density=True, volume=True))

    # equilibrate at this pressure
    simulation.step(n_steps)



    simulation.reporters.append(app.PDBReporter(lastframe_name, 1))
    simulation.step(1)
