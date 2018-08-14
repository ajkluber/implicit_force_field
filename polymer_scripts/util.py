import os
import numpy as np

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app
import mdtraj as md

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

def output_filenames(name, traj_idx):
    min_name = name + "_min_{}.pdb".format(traj_idx)
    log_name = name + "_{}.log".format(traj_idx)
    traj_name = name + "_traj_{}.dcd".format(traj_idx)
    lastframe_name = name + "_fin_{}.pdb".format(traj_idx)
    return min_name, log_name, traj_name, lastframe_name

def get_starting_coordinates(name, traj_idx):

    # get initial configuration
    if traj_idx == 1:
        print "starting new simulation from: " + name + "_min.pdb"
        pdb = app.PDBFile(name + "_min.pdb")
        minimize = True
    else:
        minimize = False
        if os.path.exists(name + "_fin_{}.pdb".format(traj_idx - 1)):
            print "extending from " + name + "_fin_{}.pdb".format(traj_idx - 1)
            pdb = app.PDBFile(name + "_fin_{}.pdb".format(traj_idx - 1))
        elif os.path.exists(name + "_traj_{}.dcd".format(traj_idx - 1)) and os.path.exists(name + "_min.pdb"):
            print "extending from final frame of " + name + "_traj_{}.pdb".format(traj_idx - 1)
            import mdtraj as md
            traj = md.load(name + "_traj_{}.dcd".format(traj_idx - 1), top=name + "_min.pdb")
            traj[-1].save_pdb(name + "_fin_{}.pdb".format(traj_idx - 1))
            pdb = app.PDBFile(name + "_fin_{}.pdb".format(traj_idx - 1))
        else:
            raise IOError("No structure to start from!")

    topology = pdb.topology
    positions = pdb.positions
    return topology, positions

def add_elements(mass_slv, mass_ply):
    # we define our coarse-grain beads as additional elements in OpenMM.
    # should only be called once!
    app.element.solvent = app.element.Element(201, "Solvent", "Sv", mass_slv)
    app.element.polymer = app.element.Element(200, "Polymer", "Pl", mass_ply)


def LJslv_params(eps_slv_mag):
    sigma_slv = 0.31*unit.nanometer
    eps_slv = eps_slv_mag*unit.kilojoule_per_mole
    mass_slv = 18.*unit.amu
    return eps_slv, sigma_slv, mass_slv

def get_Hdir(name, ply_potential, slv_potential, eps_ply_mag, eps_slv_mag):
    Hdir = os.getcwd() + "/{}_{}_{}slv".format(name, ply_potential, slv_potential)
    if ply_potential == "LJ": 
        if slv_potential == "LJ": 
            Hdir += "/eps_ply_{:.2f}_eps_slv_{:.2f}".format(eps_ply_mag, eps_slv_mag)
        else:
            Hdir += "/eps_ply_{:.2f}".format(eps_ply_mag)
    else:
        if slv_potential == "LJ": 
            Hdir += "/eps_slv_{:.2f}".format(eps_ply_mag, eps_slv_mag)
    return Hdir
