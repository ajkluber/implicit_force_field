import os
import time
import numpy as np
import sympy
import matplotlib.pyplot as plt

import mdtraj as md
import simtk.unit as unit

import simulation.openmm as sop

def compare_calculated_forces_to_simulation():
    print("calculating trajectory derivatives...")
    starttime = time.time()
    total_err = 0
    for chunk in md.iterload(trajfile, top=topfile):
        xyz_flat = np.reshape(chunk.xyz, (chunk.n_frames, n_dim))
        D_ikl = np.zeros((chunk.n_frames, n_basis_deriv), float)

        # calculate forces
        for i in range(len(dU_all)):
            D_ikl[:,i] = dU_all[i](*xyz_flat[:,dU_idxs[i]].T)

        # multiply force field constants through each row
        for i in range(D_ikl.shape[0]):
            D_ikl[i,:] *= c
        Y_il = (xyz_flat[s_frames:,:] - xyz_flat[:-s_frames,:])/s
        err = np.sum((Y_il - np.dot(D_ikl, fk_with_dxi)[:-s_frames])**2)
        total_err += err
        break
    stoptime = time.time()
    runmin = (stoptime - starttime)/60.
    print("calculation took: {} min".format(runmin))

    # compare computed forces to real forces
    f_calc = np.dot(D_ikl, fk_with_dxi)

    forcefile = "c25_forces_1.dat"
    f_sim = np.loadtxt(forcefile)

    plt.figure()
    plt.plot(f_sim[:f_calc.shape[0]].ravel(), f_calc.ravel(), '.')
    xmin, xmax = plt.xlim()
    plt.plot([xmin, xmax], [xmin, xmax], 'k--')
    plt.xlabel(r"$f_{sim}$ (kJ/mol nm)")
    plt.ylabel(r"$f_{calc}$ (kJ/mol nm)")
    plt.savefig("force_sim_vs_force_calc.pdf")
    plt.savefig("force_sim_vs_force_calc.png")

def calculate_energy_terms():
    # define energy functions
    U_bond_sym = []
    for i in range(n_beads - 1):
        U_bond_sym.append(one_half*(rij_sym[i][0] - r0_nm)**2)

    U_ang_sym = []
    for i in range(n_beads - 2):
        U_ang_sym.append(one_half*(theta_ijk_sym[i] - theta0_rad)**2)

    U_ang = []
    for i in range(n_beads - 2):
        x1, y1, z1 = xyz_sym[i]
        x2, y2, z2 = xyz_sym[i + 1]
        x3, y3, z3 = xyz_sym[i + 2]
        args = (x1, y1, z1, x2, y2, z2, x3, y3, z3)
        U_ang.append(sympy.lambdify(args, U_ang_sym[i], modules="numpy"))

    bond_idxs = []
    for i in range(n_beads - 1):
        bond_idxs.append([i, i + 1])
    bond_idxs = np.array(bond_idxs)

    ang_idxs = []
    for i in range(n_beads - 2):
        ang_idxs.append([i, i + 1, i + 2])
    ang_idxs = np.array(ang_idxs)

    pairs_idxs = []
    for i in range(n_beads - 3):
        for j in range(i + 4, n_beads):
            pairs_idxs.append([i, j])
    pairs_idxs = np.array(pairs_idxs)

    for chunk in md.iterload(trajfile, top=topfile, chunk=1000):
        bond_rij = md.compute_distances(chunk, bond_idxs)
        pair_rij = md.compute_distances(chunk, pairs_idxs)
        theta = md.compute_angles(chunk, ang_idxs)
        break


    # simulation observables trajectory
    traj = md.load(trajfile, top=topfile)
    #bond_rij = md.compute_distances(traj, np.array([[4, 5]]))
    bond_rij = md.compute_distances(traj, bond_idxs)
    pair_rij = md.compute_distances(traj, pairs_idxs)
    theta = md.compute_angles(traj, ang_idxs)

    # total energy calculated
    U_pair = np.sum(sop.tabulated.WCA(pair_rij, sigma_ply_nm, eps_ply_kj), axis=1)
    U_bond = np.sum(0.5*kb_kj*(bond_rij - r0_nm)**2, axis=1)
    U_ang = np.sum(0.5*ka_kj*(theta - theta0_rad)**2, axis=1)
    U_tot = U_pair + U_bond + U_ang
    
    # total energy reported from simulation
    E_tot = np.loadtxt("c25_1.log", usecols=(1,), delimiter=",", skiprows=1)

    # compare total energy in simulation and python code

    #x1, y1, z1 = chunk.xyz[:,0,:].T
    #x2, y2, z2 = chunk.xyz[:,1,:].T
    #x3, y3, z3 = chunk.xyz[:,2,:].T
    #temp = U_ang[0](x1, y1, z1, x2, y2, z2, x3, y3, z3) 

    #temp_U = sympy.lambdify(theta_ijk_sym[0], U_ang_sym[0])

    #theta_ijk = sympy.symbol("theta_ijk")
    #
    #
    #args = (xyz_sym[0][0], xyz_sym[0][1], xyz_sym[0][2], 
    #        xyz_sym[1][0], xyz_sym[1][1], xyz_sym[1][2], 
    #        xyz_sym[2][0], xyz_sym[2][1], xyz_sym[2][2])

    #dtheta_ijk_dxi = sympy.lambdify(args, -theta_ijk_sym[0].diff(xyz_sym[0][0]), modules="numpy")
    ##theta_ijk_sym.append(sympy.acos((r_ij**2 + r_jk**2 - r_ik**2)/(2*r_ij*r_jk)))
    ##U_ang.append(sympy.lambdify(args, U_ang_sym[i], modules="numpy"))

if __name__ == "__main__":
    n_beads = 25 
    n_dim = 3*n_beads

    #topfile = "c25_min_ply_1.pdb"
    #trajfile = "c25_traj_ply_1.dcd"
    topfile = "c25_min_1.pdb"
    trajfile = "c25_traj_1.dcd"

    #s_frames = 50
    s_frames = 10
    dt_frame = 0.2
    s = dt_frame*s_frames

    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    r0, kb, theta0, ka = bonded_params 

    # remove units from paramters
    sigma_ply_nm = sigma_ply/unit.nanometer
    r0_wca_nm = sigma_ply_nm*(2**(1./6))
    eps_ply_kj = eps_ply/unit.kilojoule_per_mole
    kb_kj = kb/(unit.kilojoule_per_mole/(unit.nanometer**2))
    ka_kj = (ka/(unit.kilojoule_per_mole/(unit.radian**2)))
    theta0_rad = theta0/unit.radian
    r0_nm = r0/unit.nanometer
