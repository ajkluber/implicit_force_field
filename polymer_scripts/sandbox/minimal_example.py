import sys

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

class ForceReporter(object):
    def __init__(self, filename, reportInterval):
        self._out = open(filename, 'w', buffering=1)
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(unit.kilojoules/unit.mole/unit.nanometer)
        f_string = ""
        for f in forces:
            f_string += '{} {} {} '.format(f[0], f[1], f[2])
        f_string = f_string[:-1] + '\n'
        self._out.write(f_string)

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
    #n_beads = 5
    #n_beads = 4
    #n_beads = 3
    name = "c" + str(n_beads)
    topname = name + "_min.pdb"

    T = 300
    temperature = T*unit.kelvin
    collision_rate = 1.0/unit.picosecond
    timestep = 0.002*unit.picosecond
    cutoff = 0.9*unit.nanometers
    vdwCutoff = 0.9*unit.nanometers
    r_switch = 0.7*unit.nanometers

    n_steps = 1000
    nsteps_out = 1

    traj_idx = 1
    min_name = name + "_min_{}.pdb".format(traj_idx)
    log_name = name + "_{}.log".format(traj_idx)
    traj_name = name + "_traj_{}.dcd".format(traj_idx)
    lastframe_name = name + "_fin_{}.pdb".format(traj_idx)
    force_name = name + "_forces_{}.dat".format(traj_idx) 

    # need to define element types before reading in the topology
    app.element.polymer = app.element.Element(200, "Polymer", "Pl", 37*unit.amu)

    pdb = app.PDBFile(topname)
    topology = pdb.topology
    positions = pdb.positions

    templates = template_dict(topology, n_beads)

    ff_filename = "ff_c{}.xml".format(n_beads)

    # Write ff file
    #import simulation.openmm as sop
    #sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()

    #ff_kwargs = {}
    #ff_kwargs["sigma_ply"] = sigma_ply
    #ff_kwargs["eps_ply"] = eps_ply
    #ff_kwargs["mass_ply"] = mass_ply

    #sop.build_ff.polymer_in_solvent(n_beads, "LJ6", "no",
    #        saveas=ff_filename, lj14scale=1, **ff_kwargs)

    forcefield = app.ForceField(ff_filename)

    system = forcefield.createSystem(topology,
        nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff,
        switchDistance=r_switch, ignoreExternalBonds=True, residueTemplates=templates)

    integrator = omm.LangevinIntegrator(temperature, collision_rate, timestep)

    platform = omm.Platform.getPlatformByName('CPU') 

    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    simulation.reporters.append(app.DCDReporter(traj_name, nsteps_out))
    simulation.reporters.append(app.StateDataReporter(log_name, nsteps_out,
        step=True, potentialEnergy=True))

    simulation.reporters.append(ForceReporter(force_name, nsteps_out))

    simulation.step(n_steps)

