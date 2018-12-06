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
    n_beads = 25
    #n_beads = 2
    name = "c" + str(n_beads)
    #topname = "c25_nosolv_min.pdb"
    topname = "c25_min.pdb"
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
    #platform = omm.Platform.getPlatformByName('CPU') 
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
    ff_kwargs["mass_ply"] = mass_ply
    ff_kwargs["eps_slv"] = eps_slv
    ff_kwargs["sigma_slv"] = sigma_slv
    ff_kwargs["mass_slv"] = mass_slv

    raise SystemExit

    #sop.build_ff.polymer_in_solvent(n_beads, "INVR12", "LJ",
    #        saveas=ff_filename, **ff_kwargs)
    sop.build_ff.polymer_in_solvent(n_beads, "LJ6", "no",
            saveas=ff_filename, **ff_kwargs)

    forcefield = app.ForceField(ff_filename)

    system = forcefield.createSystem(topology,
        nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff,
        switchDistance=r_switch, ignoreExternalBonds=True, residueTemplates=templates)

    for i in range(system.getNumForces()):
        force = system.getForce(i)
        force.setForceGroup(i)

    integrator = omm.LangevinIntegrator(temperature, collision_rate, timestep)

    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)


    simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
    simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
        step=True, potentialEnergy=True))

    simulation.reporters.append(sop.additional_reporters.ForceReporter(name + "_forces_{}.dat".format(traj_idx), nsteps_out))

    simulation.step(100*nsteps_out + 5)

    Eb = []
    fsim = []
    for i in range(100):
        simulation.step(nsteps_out)
        state = simulation.context.getState(getPositions=True, getEnergy=True, getForces=True, groups={0})

        temp_x = state.getPositions()/unit.nanometer
        temp_Eb = state.getPotentialEnergy()/unit.kilojoule_per_mole
        temp_f = state.getForces()/(unit.kilojoule_per_mole/unit.nanometer)
        Eb.append(temp_Eb)
    Eb = np.array(Eb)

    xyz_flat = np.concatenate(temp_x)

    raise SystemExit
    print "{:>10s}   {:>10s}   {:>10s}".format("Eb","Ea","Ew")
    Eb = []
    Ea = []
    Ew = []
    for i in range(100):
        simulation.step(nsteps_out)

        state = simulation.context.getState(getEnergy=True, groups={0})
        temp_Eb = state.getPotentialEnergy()/unit.kilojoule_per_mole
        Eb.append(temp_Eb)

        state = simulation.context.getState(getEnergy=True, groups={1})
        temp_Ea = state.getPotentialEnergy()/unit.kilojoule_per_mole
        Ea.append(temp_Ea)

        state = simulation.context.getState(getEnergy=True, groups={2})
        temp_Ew = state.getPotentialEnergy()/unit.kilojoule_per_mole
        Ew.append(temp_Ew)

        print "{:10.5f}   {:10.5f}   {:10.5f}".format(temp_Eb, temp_Ea, temp_Ew)
    Eb = np.array(Eb)
    Ea = np.array(Ea)
    Ew = np.array(Ew)

    raise SystemExit

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
    Ucg.harmonic_bond_potentials(r0_nm, scale_factor=kb_kj, fixed=True)
    Ucg.harmonic_angle_potentials(theta0_rad, scale_factor=ka_kj, fixed=True)
    #Ucg.inverse_r12_potentials(sigma_ply_nm, scale_factor=eps_ply_kj, fixed=True)
    Ucg.LJ6_potentials(sigma_ply_nm, eps_ply_kj, scale_factor=eps_ply_kj, fixed=True)

    traj = md.load(traj_name, top=topname)
    Ecg_terms = Ucg.calculate_potential_terms(traj)
    Eb_cg, Ea_cg, Ew_cg = Ecg_terms[0]
    Ecg_tot = Eb_cg + Ea_cg + Ew_cg

    Esim_tot = Eb + Ea + Ew
    print np.corrcoef(Ecg_tot, Esim_tot)[0,1]

    E1 = [Eb, Ea, Ew, Esim_tot]
    E2 = [Eb_cg, Ea_cg, Ew_cg, Ecg_tot]
    labels = ["bonds", "angles", "ex vol", "Total"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    idx = 0
    for i in range(2):
        for j in range(2):
            ax = axes[i,j]
            x = E1[idx]
            y = E2[idx]
            crr = np.corrcoef(x, y)[0,1]
            xmin = np.min([ np.min(x), np.min(y) ])
            xmax = np.max([ np.max(x), np.max(y) ])

            ax.annotate("{:.4f}".format(crr), xy=(0,0), xytext=(0.6, 0.2),
                    textcoords="axes fraction", xycoords="axes fraction", fontsize=15)
            ax.annotate(labels[idx], xy=(0,0), xytext=(0.2, 0.8),
                    textcoords="axes fraction", xycoords="axes fraction", fontsize=15)

            ax.plot([xmin, xmax], [xmin, xmax], 'k--')
            ax.plot(x, y, '.')

            if i == 1:
                ax.set_xlabel("sim")
            if j == 0:
                ax.set_ylabel("calc")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(xmin, xmax)
            idx += 1

    fig.savefig("compare_terms_2.pdf")
    fig.savefig("compare_terms_2.png")
