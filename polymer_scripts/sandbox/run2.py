import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import mdtraj as md

import simulation.openmm as sop
import implicit_force_field as iff
#import implicit_force_field.polymer_scripts.util as util

def template_dict(topology, n_beads):
    # tell OpenMM which residues are which in the forcefield. Otherwise
    # OpenMM is thrown by all residues having matching sets of atoms. 
    templates = {}
    idx = 1
    for res in topology.residues():
        templates[res] = "PL" + str(idx)
        if idx >= n_beads:
            break
        idx += 1
    return templates


if __name__ == "__main__":
    n_beads = 5
    #n_beads = 2
    name = "c" + str(n_beads)
    #topname = "c25_nosolv_min.pdb"
    topname = "c5_min.pdb"
    #topname = "c2_min.pdb"

    T = 300
    temperature = T*unit.kelvin
    dynamics = "Langevin"
    ensemble = "NVT"
    minimize = False
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond
    #collision_rate = 1.0/unit.picosecond
    #timestep = 0.002*unit.picosecond
    cutoff = 0.9*unit.nanometers
    vdwCutoff = 0.9*unit.nanometers
    r_switch = 0.7*unit.nanometers

    n_steps = 1000
    nsteps_out = 1

    #properties = {'DeviceIndex': '0'}
    platform = omm.Platform.getPlatformByName('CPU') 
    #platform = omm.Platform.getPlatformByName('CUDA') 

    traj_idx = 1
    min_name = name + "_min_{}.pdb".format(traj_idx)
    log_name = name + "_{}.log".format(traj_idx)
    traj_name = name + "_traj_{}.dcd".format(traj_idx)
    lastframe_name = name + "_fin_{}.pdb".format(traj_idx)

    # need to define element types before reading in the topology
    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    eps_slv, sigma_slv, B, r0, Delta, mass_slv = sop.build_ff.CS_water_params()
    app.element.polymer = app.element.Element(200, "Polymer", "Pl", mass_ply)
    app.element.solvent = app.element.Element(201, "Solvent", "Sv", mass_slv)

    pdb = app.PDBFile(topname)
    topology = pdb.topology
    positions = pdb.positions
    #templates = util.template_dict(topology, n_beads)
    templates = sop.util.template_dict(topology, n_beads)
    #templates = template_dict(topology, n_beads)

    ff_filename = "ff_c{}.xml".format(n_beads)

    ff_kwargs = {}
    ff_kwargs["sigma_ply"] = sigma_ply
    ff_kwargs["eps_ply"] = eps_ply
    ff_kwargs["mass_ply"] = mass_ply
    ff_kwargs["eps_slv"] = eps_slv
    ff_kwargs["sigma_slv"] = sigma_slv
    ff_kwargs["mass_slv"] = mass_slv

    #sop.build_ff.polymer_in_solvent(n_beads, "INVR12", "LJ",
    #        saveas=ff_filename, **ff_kwargs)
    sop.build_ff.polymer_in_solvent(n_beads, "LJ6", "no",
            saveas=ff_filename, **ff_kwargs)

    forcefield = app.ForceField(ff_filename)

    system = forcefield.createSystem(topology,
        nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff,
        switchDistance=r_switch, ignoreExternalBonds=True, residueTemplates=templates)

    #for i in range(system.getNumForces()):
    #    force = system.getForce(i)
    #    force.setForceGroup(i)

    integrator = omm.LangevinIntegrator(temperature, collision_rate, timestep)

    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
    simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
        step=True, potentialEnergy=True))

    simulation.reporters.append(sop.additional_reporters.ForceReporter(name + "_forces_{}.dat".format(traj_idx), nsteps_out))

    simulation.step(100*nsteps_out + 5)

    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    r0, kb, theta0, ka = bonded_params

    sigma_ply_nm = sigma_ply/unit.nanometer
    r0_wca_nm = sigma_ply_nm*(2**(1./6))
    eps_ply_kj = eps_ply/unit.kilojoule_per_mole
    kb_kj = kb/(unit.kilojoule_per_mole/(unit.nanometer**2))
    ka_kj = (ka/(unit.kilojoule_per_mole/(unit.radian**2)))
    theta0_rad = theta0/unit.radian
    r0_nm = r0/unit.nanometer

    Ucg = iff.basis_library.PolymerModel(n_beads)
    Ucg.harmonic_bond_potentials(r0_nm, scale_factor=kb_kj)
    Ucg.harmonic_angle_potentials(theta0_rad, scale_factor=ka_kj)
    #Ucg.inverse_r12_potentials(sigma_ply_nm, scale_factor=eps_ply_kj)
    Ucg.LJ6_potentials(sigma_ply_nm, eps_ply_kj, scale_factor=1)

    traj = md.load(traj_name, top=topname)
    Eterms = Ucg.calculate_potential_terms(traj)
    Etot_cg = np.array(Eterms[1]).sum(axis=1)

    G = Ucg.calculate_parametric_forces(traj)

    fsim = np.loadtxt(name + "_forces_{}.dat".format(traj_idx))
    frav = fsim.ravel()

    c_soln = np.linalg.lstsq(G[:len(frav)],frav)
