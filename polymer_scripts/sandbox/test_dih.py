import os
import sympy
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

if __name__ == "__main__":
    n_beads = 25
    #n_beads = 2
    name = "c" + str(n_beads)
    topname = "c25_nosolv_min.pdb"

    traj_idx = 2
    min_name = name + "_min_{}.pdb".format(traj_idx)
    log_name = name + "_{}.log".format(traj_idx)
    traj_name = name + "_traj_{}.dcd".format(traj_idx)
    lastframe_name = name + "_fin_{}.pdb".format(traj_idx)

    ply_idxs = np.arange(25)

    dih_idxs = []
    for i in range(len(ply_idxs) - 3):
        idx = ply_idxs[i]
        dih_idxs.append([idx, idx + 1, idx + 2, idx + 3])
    dih_idxs = np.array(dih_idxs)
    n_ang = len(dih_idxs)

    # need to define element types before reading in the topology
    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    eps_slv, sigma_slv, B, r0, Delta, mass_slv = sop.build_ff.CS_water_params()
    app.element.polymer = app.element.Element(200, "Polymer", "Pl", mass_ply)
    app.element.solvent = app.element.Element(201, "Solvent", "Sv", mass_slv)

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
    Ucg.inverse_r12_potentials(sigma_ply_nm, scale_factor=eps_ply_kj, fixed=True)

    traj = md.load("c25_traj_cent_1.dcd", top="c25_nosolv_min.pdb")
    phi_sim = md.compute_dihedrals(traj, dih_idxs)

    coords = traj.xyz[:,(0,1,2,3)]
    xyz_flat = np.reshape(coords, (traj.n_frames, 12))

    phi_fun = sympy.lambdify(Ucg.phi_ijkl_args, Ucg.phi_ijkl_sym, modules="numpy")
    phi_fun2 = sympy.lambdify(Ucg.phi_ijkl_args, Ucg.phi_ijkl_sym_mdtraj, modules="numpy")

    phi_cg = phi_fun(*xyz_flat.T)
    phi_cg2 = phi_fun2(*xyz_flat.T)


    plt.figure()
    plt.plot(phi_sim[:,0], phi_cg, 'ro', label=r"$\arccos$")
    plt.plot(phi_sim[:,0], phi_cg2, 'bo', label=r"$\arctan$")
    plt.legend(loc=5)
    plt.xlabel("MDTraj")
    plt.ylabel("Sympy")
    #plt.xlim(-np.pi, np.pi)
    #plt.ylim(-np.pi, np.pi)
    plt.savefig("phi_comp.pdf")
    plt.savefig("phi_comp.png")


