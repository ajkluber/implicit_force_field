import os
import time
import numpy as np
import sympy
import matplotlib.pyplot as plt

import mdtraj as md
import simtk.unit as unit

import simulation.openmm as sop

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

    one_half = sympy.Rational(1,2)

    # define all variables of the system symbolically
    print("making variables...")
    x_idxs = []
    xyz_sym = []
    for i in range(n_beads):
        x_i = sympy.symbols('x' + str(i + 1))
        y_i = sympy.symbols('y' + str(i + 1))
        z_i = sympy.symbols('z' + str(i + 1))
        xyz_sym.append([x_i, y_i, z_i])

    rij_sym = []
    for i in range(n_beads - 1):
        x1, y1, z1 = xyz_sym[i]
        ri_list = []
        for j in range(i + 1, n_beads):
            x2, y2, z2 = xyz_sym[j]
            ri_list.append(sympy.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2))
        rij_sym.append(ri_list)

    theta_ijk_sym = []
    for i in range(n_beads - 2):
        r_ij = rij_sym[i][0]
        r_jk = rij_sym[i + 1][0]
        r_ik = rij_sym[i][1]
        theta_ijk_sym.append(sympy.acos((r_ij**2 + r_jk**2 - r_ik**2)/(2*r_ij*r_jk)))

    # calculate gradient with respect to each coordinate
    # for each function have list of participating coordinates.
    print("making derivative lambda functions...")
    idxs_for_dfk_dxi = [] 
    col_idx = 0

    # How do we reduce the number of parameters?
    #  - Repeated interactions. E.g. all bonds interact with the same basis
    #    function. 24 params -> 1

    dU_bond = []
    dbond_idxs = []
    dbond_deriv_idxs = []
    for i in range(n_beads - 1):
        x1, y1, z1 = xyz_sym[i]
        x2, y2, z2 = xyz_sym[i + 1]
        args = (x1, y1, z1, x2, y2, z2)
        bond_func = one_half*(rij_sym[i][0] - r0_nm)**2
        #bond_func = (one_half/kb_kj)*(rij_sym[i][0] - r0_nm)**2 # scaled
        xi_idxs = np.arange(6) + i*3
        temp_dfk_dxi = []

        for n in range(len(args)):
            # take derivative w.r.t. argument n
            d_bond_func = -bond_func.diff(args[n])
            dU_bond.append(sympy.lambdify(args, d_bond_func, modules="numpy"))
            dbond_idxs.append(xi_idxs)
            dbond_deriv_idxs.append(xi_idxs[n])    # this is what the derivative is w.r.t.

            temp_dfk_dxi.append((xi_idxs[n], col_idx))
            col_idx += 1
        idxs_for_dfk_dxi.append(temp_dfk_dxi)

    # angle potential
    dU_ang = []
    dang_idxs = []
    dang_deriv_idxs = []
    for i in range(n_beads - 2):
        x1, y1, z1 = xyz_sym[i]
        x2, y2, z2 = xyz_sym[i + 1]
        x3, y3, z3 = xyz_sym[i + 2]
        args = (x1, y1, z1, x2, y2, z2, x3, y3, z3)
        ang_func = one_half*(theta_ijk_sym[i] - theta0_rad)**2
        #ang_func = (one_half/ka_kj)*(theta_ijk_sym[i] - theta0_rad)**2  # scaled
        xi_idxs = np.arange(9) + i*3
        temp_dfk_dxi = []

        for n in range(len(args)):
            d_ang_func = -ang_func.diff(args[n])
            dU_ang.append(sympy.lambdify(args, d_ang_func, modules="numpy"))
            dang_idxs.append(xi_idxs)
            dang_deriv_idxs.append(xi_idxs[n])

            temp_dfk_dxi.append((xi_idxs[n], col_idx))
            col_idx += 1
        idxs_for_dfk_dxi.append(temp_dfk_dxi)

    # pairwise potential
    #bond_cutoff = 3
    #dU_pair = []
    #dpair_idxs = []
    #dpair_deriv_idxs = []
    #for i in range(n_beads - bond_cutoff - 1):
    #    x1, y1, z1 = xyz_sym[i]
    #    idxs1 = np.arange(3) + i*3
    #    for j in range(i + bond_cutoff + 1, n_beads):
    #        x2, y2, z2 = xyz_sym[j]
    #        args = (x1, y1, z1, x2, y2, z2)
    #        idxs2 = np.arange(3) + j*3
    #        xi_idxs = np.concatenate([idxs1, idxs2])

    #        neighbor_idx = j - i - 1 
    #        r = rij_sym[i][neighbor_idx]
    #        # WCA function. Have to use a quickly switching tanh function
    #        # instead of Heaviside step function to ensure differentiability
    #        #pair_func = one_half*(sympy.tanh(400*(r0_wca_nm - r)) + 1)*(4*((sigma_ply_nm/r)**12 - (sigma_ply_nm/r)**6) + 1)
    #        pair_func = (1./eps_ply_kj)*one_half*(sympy.tanh(400*(r0_wca_nm - r)) + 1)*(4*((sigma_ply_nm/r)**12 - (sigma_ply_nm/r)**6) + 1) # scaled

    #        temp_dfk_dxi = []
    #            
    #        for n in range(len(args)):
    #            d_pair_func = -pair_func.diff(args[n])
    #            dU_pair.append(sympy.lambdify(args, d_pair_func, modules="numpy"))
    #            dpair_idxs.append(xi_idxs)
    #            dpair_deriv_idxs.append(xi_idxs[n])

    #            temp_dfk_dxi.append((xi_idxs[n], col_idx))
    #            col_idx += 1
    #        idxs_for_dfk_dxi.append(temp_dfk_dxi)

    c = np.concatenate([ kb_kj*np.ones(len(dU_bond)), ka_kj*np.ones(len(dU_ang)) ])
    #c = np.concatenate([ kb_kj*np.ones(len(dU_bond)), ka_kj*np.ones(len(dU_ang)), eps_ply_kj*np.ones(len(dU_pair)) ])

    #dU_all = dU_bond + dU_ang + dU_pair
    #dU_idxs = dbond_idxs + dang_idxs + dpair_idxs
    #dU_deriv_idxs = np.array(dbond_deriv_idxs + dang_deriv_idxs + dpair_deriv_idxs)
    dU_all = dU_bond + dU_ang
    dU_idxs = dbond_idxs + dang_idxs
    dU_deriv_idxs = np.array(dbond_deriv_idxs + dang_deriv_idxs)

    n_basis_deriv = len(dU_all)
    n_basis = len(idxs_for_dfk_dxi)

    # which derivatives are with respect to x_i?
    fk_with_dxi = np.zeros((n_basis_deriv, n_dim), int)
    for i in range(n_dim):
        fk_with_dxi[np.argwhere(dU_deriv_idxs == i)[:,0],i] = 1
    fk_with_dxi = fk_with_dxi.astype(bool)

    print("getting column indices for derivatives... ")
    fk_and_fk_prime_with_dxi = {}
    for i in range(n_basis):
        for j in range(i, n_basis):
            idxs1 = [ x[0] for x in idxs_for_dfk_dxi[i] ]
            idxs2 = [ x[0] for x in idxs_for_dfk_dxi[j] ]

            dxi_in_both = np.intersect1d(idxs1, idxs2)
            if len(dxi_in_both) > 0:
                in_both_col_idxs = []
                # if functions shared some arguments
                # determine column indices of all derivatives that they share
                for n in range(len(dxi_in_both)):
                    xi_idx1 = idxs_for_dfk_dxi[i][idxs1.index(dxi_in_both[n])][0]
                    xi_idx2 = idxs_for_dfk_dxi[j][idxs2.index(dxi_in_both[n])][0]
                    col_idx1 = idxs_for_dfk_dxi[i][idxs1.index(dxi_in_both[n])][1]
                    col_idx2 = idxs_for_dfk_dxi[j][idxs2.index(dxi_in_both[n])][1]
                    in_both_col_idxs.append(((xi_idx1, xi_idx2), (col_idx1, col_idx2)))
                fk_and_fk_prime_with_dxi[str(i) + "," + str(j)] = in_both_col_idxs

    n_frames_tot = 0
    for chunk in md.iterload(trajfile, top=topfile):
        n_frames_tot += chunk.n_frames
    n_frames_tot = float(n_frames_tot)

    #print("calculating trajectory derivatives...")
    #starttime = time.time()
    #D_T_D = np.zeros((n_basis, n_basis), float)
    #D_T_Y = np.zeros(n_basis, float) 
    #iteration_idx = 0
    #total_n_iters = int(np.round(n_frames_tot/1000))
    #for chunk in md.iterload(trajfile, top=topfile, chunk=1000):
    #    print("  ({}/{})".format(iteration_idx + 1, total_n_iters))
    #    xyz_flat = np.reshape(chunk.xyz, (chunk.n_frames, n_dim))
    #    D_ikl = np.zeros((chunk.n_frames, n_basis_deriv), float)

    #    # calculate forces
    #    for i in range(len(dU_all)):
    #        D_ikl[:,i] = dU_all[i](*xyz_flat[:,dU_idxs[i]].T)

    #    # calculate drift
    #    Y_il = (xyz_flat[s_frames:,:] - xyz_flat[:-s_frames,:])/s

    #    # calculate product of forces
    #    for kkprime, xi_and_col_idxs in fk_and_fk_prime_with_dxi.iteritems():
    #        k_idx, k_prime_idx = map(int, kkprime.split(","))

    #        dtd_value = 0.
    #        dty_value = 0
    #        for m in range(len(xi_and_col_idxs)):
    #            x1_idx, x2_idx = xi_and_col_idxs[0] 
    #            col1, col2 = xi_and_col_idxs[1]

    #            # multiple dfk_dxi times dfkprime_dxi
    #            dtd_value += np.sum(D_ikl[:, col1]*D_ikl[:,col2])

    #            # multiple dfk_dxi times dxi (drift)
    #            dty_value += np.sum(D_ikl[:-s_frames, col1]*Y_il[:,x1_idx])

    #        #D_T_D[k_idx, k_prime_idx] = dtd_value/n_frames_tot
    #        D_T_D[k_idx, k_prime_idx] = dtd_value
    #        if k_idx != k_prime_idx:
    #            #D_T_D[k_prime_idx, k_prime_idx] = dtd_value/n_frames_tot
    #            D_T_D[k_prime_idx, k_prime_idx] = dtd_value
    #        #D_T_Y[k_idx] = dty_value/n_frames_tot
    #        D_T_Y[k_idx] = dty_value

    #    iteration_idx += 1

    #stoptime = time.time()
    #runmin = (stoptime - starttime)/60.
    #print("calculation took: {} min".format(runmin)

    print("calculating trajectory derivatives...")
    starttime = time.time()
    avgD = np.zeros((n_basis, n_dim), float)
    avgY = np.zeros(n_basis, float) 
    iteration_idx = 0
    total_n_iters = int(np.round(n_frames_tot/1000))
    for chunk in md.iterload(trajfile, top=topfile, chunk=1000):
        print("  ({}/{})".format(iteration_idx + 1, total_n_iters))
        xyz_flat = np.reshape(chunk.xyz, (chunk.n_frames, n_dim))
        D_ikl = np.zeros((chunk.n_frames, n_basis_deriv), float)

        # calculate forces
        for i in range(len(dU_all)):
            D_ikl[:,i] = dU_all[i](*xyz_flat[:,dU_idxs[i]].T)
        sumD = np.sum(D_ikl, axis=0)/n_frames_tot
        # assign columns to 

        # calculate drift
        Y_il = (xyz_flat[s_frames:,:] - xyz_flat[:-s_frames,:])/s
        avgY += np.sum(Y_il, axis=0)/n_frames_tot)

        iteration_idx += 1

    stoptime = time.time()
    runmin = (stoptime - starttime)/60.
    print("calculation took: {} min".format(runmin))

    raise SystemExit
    q = np.diag(D_T_D)
    cond_D = np.copy(D_T_D)
    for i in range(len(cond_D)):
        cond_D[i,:] /= q 
    for j in range(len(cond_D)):
        cond_D[:,j] /= q 

    cond_Y = np.copy(D_T_Y)/q

    u_eig, sing_vals, v_eig = np.linalg.svd(cond_D)


    u_eig, sing_vals, v_eig = np.linalg.svd(D_T_D)
    #cut_idx = 240
    cut_idx = 6
    rcond = sing_vals[cut_idx]/sing_vals[cut_idx]
    rcond = 1e-6

    #inv_s = np.zeros((n_basis, n_basis))
    #inv_s[:cut_idx,:cut_idx] = np.diag(1./sing_vals[:cut_idx])
    #cond_DTD = np.dot(u_eig, np.dot(inv_s, v_eig))

    #monroe_penrose = np.dot(v_eig.T, np.dot(inv_s, u_eig.T))
    #c_soln = np.dot(monroe_penrose, D_T_Y)

    fig, axes = plt.subplots(10,1, figsize=(10, 40))
    for i in range(10):
        rcond = sing_vals[i + 1]/sing_vals[0]
        #c_soln, residuals, rank, sing_vals = np.linalg.lstsq(D_T_D, D_T_Y, rcond=rcond)
        c_soln, residuals, rank, sing_vals = np.linalg.lstsq(cond_D, cond_Y, rcond=rcond)
        axes[i].plot(q*c_soln, 'o')
    plt.show()









