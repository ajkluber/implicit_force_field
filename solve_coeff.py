import os
import time
import numpy as np
import sympy
import matplotlib.pyplot as plt
import scipy.interpolate
from sklearn.cross_validation import KFold

import mdtraj as md
import simtk.unit as unit

import simulation.openmm as sop

def do_force_matching():
    print "calculating trajectory derivatives..."
    starttime = time.time()
    iteration_idx = 0

    total_n_iters = int(np.round(n_frames_tot/1000))
    
    # Force-matching
    G = np.zeros((int(n_frames_tot)*n_dim, n_params), float)
    start_idx = 0
    for chunk in md.iterload(trajfile, top=topfile, chunk=1000):
        if ((iteration_idx + 1) % 10) == 0:
            print "  ({}/{})".format(iteration_idx + 1, total_n_iters)
        xyz_flat = np.reshape(chunk.xyz, (chunk.n_frames, n_dim))

        ravel_size = chunk.n_frames*n_dim

        # calculate forces
        for i in range(n_basis_deriv):
            # derivative 
            deriv_fun = dU_funcs[dU_ck[i]][dU_d_arg[i]]
            deriv = deriv_fun(*xyz_flat[:,dU_idxs[i]].T)

            # unraveled indices for xi 
            xi_ravel_idxs = start_idx + np.arange(dU_dxi[i], ravel_size, n_dim)
            G[xi_ravel_idxs, dU_ck[i]] += deriv.ravel()

        # calculate drift
        iteration_idx += 1
        start_idx += ravel_size

    # compare to real forces
    forces_sim = np.loadtxt("c25_forces_1.dat").ravel()
    c_soln, residuals, rank, sing_vals = np.linalg.lstsq(G, forces_sim)
    c_soln *= scale_factors

    c_sim = np.array([kb_kj, ka_kj, eps_ply_kj])

    percent_error = 100*(c_soln - c_sim)/c_sim

    print "Parm    True         Soln       % Err"
    print "kb    {:10.2f}  {:10.2f}  {:10.8f}".format(c_sim[0], c_soln[0], percent_error[0])
    print "ka    {:10.2f}  {:10.2f}  {:10.8f}".format(c_sim[1], c_soln[1], percent_error[1])
    print "eps   {:10.2f}  {:10.2f}  {:10.8f}".format(c_sim[2], c_soln[2], percent_error[2])

def calculate_derivatives_and_drift(trajfile, topfile, dU_funcs, dU_ck, dU_d_arg, dU_idxs, n_frames_tot, n_dim, n_params):
    starttime = time.time()
    iteration_idx = 0

    total_n_iters = int(np.round(n_frames_tot/1000))
    
    G = np.zeros((int(n_frames_tot)*n_dim, n_params), float)
    Y = np.zeros(int(n_frames_tot)*n_dim, float)

    start_idx = 0
    for chunk in md.iterload(trajfile, top=topfile, chunk=1000):
        if ((iteration_idx + 1) % 10) == 0:
            print "  ({}/{})".format(iteration_idx + 1, total_n_iters)
        xyz_flat = np.reshape(chunk.xyz, (chunk.n_frames, n_dim))

        ravel_size = (chunk.n_frames - s_frames)*n_dim

        # calculate forces
        for i in range(n_basis_deriv):
            # derivative 
            deriv_fun = dU_funcs[dU_ck[i]][dU_d_arg[i]]
            deriv = deriv_fun(*xyz_flat[:,dU_idxs[i]].T)[:-s_frames]   # derivative k dxi_idx = dU_dxi[i]

            # unraveled indices for xi 
            xi_ravel_idxs = start_idx + np.arange(dU_dxi[i], ravel_size, n_dim)
            G[xi_ravel_idxs, dU_ck[i]] += deriv.ravel()

        # calculate drift
        Y_il = (xyz_flat[s_frames:,:] - xyz_flat[:-s_frames,:])/s
        Y[start_idx:start_idx + ravel_size] = Y_il.ravel()

        iteration_idx += 1
        start_idx += ravel_size

    stoptime = time.time()
    runmin = (stoptime - starttime)/60.
    print "calculation took: {} min".format(runmin)

    G = G[:start_idx]
    Y = Y[:start_idx]
    return G, Y

def build_basis_function_database(n_beads):
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
    max_n_args = 3*3
    xyz_sym = []
    for i in range(max_n_args/3):
        x_i = sympy.symbols('x' + str(i + 1))
        y_i = sympy.symbols('y' + str(i + 1))
        z_i = sympy.symbols('z' + str(i + 1))
        xyz_sym.append([x_i, y_i, z_i])
    x1, y1, z1 = xyz_sym[0]
    x2, y2, z2 = xyz_sym[1]
    x3, y3, z3 = xyz_sym[2]

    r12_sym = sympy.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    r23_sym = sympy.sqrt((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2)
    r13_sym = sympy.sqrt((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2)

    rij_args = (x1, y1, z1, x2, y2, z2)

    theta_ijk_sym = sympy.acos((r12_sym**2 + r23_sym**2 - r13_sym**2)/(2*r12_sym*r23_sym))
    theta_ijk_args = (x1, y1, z1, x2, y2, z2, x3, y3, z3)

    # calculate gradient with respect to each coordinate
    # for each function have list of participating coordinates.

    n_params = 3
    
    # scale force functions to approximate magnitude of parameters. Then output
    # coeff will all be around 1. This reduces the condition number
    kb_scale = sympy.Rational(300000,1)
    ka_scale = sympy.Rational(500,1)
    eps_scale = sympy.Rational(10, 17)
    gauss_scale = sympy.Rational(1,10)
    scale_factors = np.array([ float(x) for x in [kb_scale.evalf(),
        ka_scale.evalf(), eps_scale.evalf(), gauss_scale.evalf()]])

    dU_bond_dxi = []
    dU_bond_ck = []
    dU_bond = []
    dU_bond_d_arg = []
    dbond_idxs = []
    for i in range(n_beads - 1):
        xi_idxs = np.arange(6) + i*3
        for n in range(len(rij_args)):
            dbond_idxs.append(xi_idxs)
            dU_bond_d_arg.append(n)
            dU_bond_dxi.append(xi_idxs[n])
            dU_bond_ck.append(0)
            if i == 0:
                # take derivative w.r.t. argument n
                bond_func = kb_scale*one_half*(r12_sym - r0_nm)**2 # scaled
                d_bond_func = -bond_func.diff(rij_args[n])
                dU_bond.append(sympy.lambdify(rij_args, d_bond_func, modules="numpy"))

    # angle potential
    dU_angle = []
    dU_angle_dxi = []
    dU_angle_ck = []
    dU_angle_d_arg = []
    dang_idxs = []
    for i in range(n_beads - 2):
        xi_idxs = np.arange(9) + i*3
        for n in range(len(theta_ijk_args)):
            dang_idxs.append(xi_idxs)
            dU_angle_dxi.append(xi_idxs[n])
            dU_angle_ck.append(1)
            dU_angle_d_arg.append(n)
            if i == 0:
                ang_func = ka_scale*one_half*(theta_ijk_sym - theta0_rad)**2  # scaled
                d_ang_func = -ang_func.diff(theta_ijk_args[n])
                dU_angle.append(sympy.lambdify(theta_ijk_args, d_ang_func, modules="numpy"))

    # pairwise potential
    bond_cutoff = 3
    dU_pair = []
    dU_pair_dxi = []
    dU_pair_ck = []
    dU_pair_d_arg = []
    dpair_idxs = []
    for i in range(n_beads - bond_cutoff - 1):
        idxs1 = np.arange(3) + i*3
        for j in range(i + bond_cutoff + 1, n_beads):
            idxs2 = np.arange(3) + j*3
            xi_idxs = np.concatenate([idxs1, idxs2])
            for n in range(len(rij_args)):
                dpair_idxs.append(xi_idxs)
                dU_pair_dxi.append(xi_idxs[n])
                dU_pair_ck.append(2)
                dU_pair_d_arg.append(n)
                if (i == 0) and (j == (bond_cutoff + 1)):
                    pair_func = eps_scale*one_half*(sympy.tanh(400*(r0_wca_nm - r12_sym)) + 1)*(4*((sigma_ply_nm/r12_sym)**12 - (sigma_ply_nm/r12_sym)**6) + 1)
                    d_pair_func = -pair_func.diff(rij_args[n])
                    dU_pair.append(sympy.lambdify(rij_args, d_pair_func, modules="numpy"))


    # create spline basis functions
    # for each pair
    #    for each spline
    #        lambdify derivative of spline
    #        assign pair to derivative

    gauss_r0 = [ sympy.Rational(3 + i,10) for i in range(10) ]
    gauss_w = sympy.Rational(1, 10)

    bond_cutoff = 3
    dU_gauss = []
    dU_gauss_dxi = []
    dU_gauss_ck = []
    dU_gauss_d_arg = []
    dgauss_idxs = []

    # add a gaussian well
    for m in range(len(gauss_r0)):
        dU_m = []
        for i in range(n_beads - bond_cutoff - 1):
            idxs1 = np.arange(3) + i*3
            for j in range(i + bond_cutoff + 1, n_beads):
                idxs2 = np.arange(3) + j*3
                xi_idxs = np.concatenate([idxs1, idxs2])

                # loop over basis functions
                for n in range(len(rij_args)):
                    dgauss_idxs.append(xi_idxs)
                    dU_gauss_dxi.append(xi_idxs[n])

                    # add 
                    dU_gauss_ck.append(3 + m)
                    dU_gauss_d_arg.append(n)
                    if (i == 0) and (j == (bond_cutoff + 1)):
                        gauss_func = -gauss_scale*sympy.exp(-one_half*((r12_sym - gauss_r0[m])/gauss_w)**2)

                        d_gauss_func = -gauss_func.diff(rij_args[n])
                        dU_m.append(sympy.lambdify(rij_args, d_gauss_func, modules="numpy"))
        dU_gauss.append(dU_m)

    dU_funcs = [dU_bond, dU_angle, dU_pair] + dU_gauss
    dU_idxs = dbond_idxs + dang_idxs + dpair_idxs + dgauss_idxs
    dU_d_arg = dU_bond_d_arg + dU_angle_d_arg + dU_pair_d_arg + dU_gauss_d_arg
    dU_dxi = dU_bond_dxi + dU_angle_dxi + dU_pair_dxi + dU_gauss_dxi
    dU_ck = dU_bond_ck + dU_angle_ck + dU_pair_ck + dU_gauss_ck

    return dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, scale_factors

if __name__ == "__main__":
    n_beads = 25 
    n_dim = 3*n_beads
    n_folds = 10
    dt_frame = 0.2

    #topfile = "c25_min_ply_1.pdb"
    #trajfile = "c25_traj_ply_1.dcd"
    topfile = "c25_min_1.pdb"
    trajfile = "c25_traj_1.dcd"

    print "building basis function database..."
    dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, scale_factors = build_basis_function_database(n_beads)
    n_basis_deriv = len(dU_dxi)
    n_params = len(dU_funcs)

    n_frames_tot = 0
    for chunk in md.iterload(trajfile, top=topfile):
        n_frames_tot += chunk.n_frames
    n_frames_tot = float(n_frames_tot)

    all_s = [1, 2, 5, 10, 20, 50, 100, 500]
    all_cv_scores = [] 
    all_c_soln = []
    for z in range(len(all_s)):
        #for z in [0]:
        s_frames = all_s[z]
        s = dt_frame*s_frames

        print "calculating trajectory derivatives..."
        G, Y = calculate_derivatives_and_drift(trajfile, topfile, dU_funcs, dU_ck, dU_d_arg, dU_idxs, n_frames_tot, n_dim, n_params)

        cv_score = 0
        temp_c_soln = []
        kf = KFold(Y.shape[0], n_folds=n_folds, shuffle=True)
        for train_idxs, test_idxs in kf:
            # cross-validation: solve regression on one part of data then test
            # it on another. Helps measure predictability.
            c_new = np.linalg.lstsq(G[train_idxs], Y[train_idxs])[0]
            y_fit_new = np.dot(G, c_new)
            cv_score += np.linalg.norm(Y[test_idxs] - y_fit_new[test_idxs], ord=2)
            temp_c_soln.append(c_new)
        all_c_soln.append(temp_c_soln)
        cv_score /= float(n_folds)
        all_cv_scores.append(cv_score)

    all_s = np.array(all_s)
    all_cv_scores = np.array(all_cv_scores)

    c_vs_s = []
    for n in range(n_params):
        c_vs_s.append(scale_factors[n]*np.array([ np.array(all_c_soln[i])[:,n] for i in range(len(all_s)) ]))

    raise SystemExit

    ylabels = [r"$k_b$", r"$k_a$", r"$\epsilon$"]

    fig, axes = plt.subplots(3, 1, figsize=(4,10))
    for i in range(n_params):
        coeff = c_vs_s[i]
        ax = axes[i]
        ax.errorbar(dt_frame*all_s, np.mean(coeff, axis=1), yerr=np.std(coeff, axis=1))
        ax.set_ylabel(ylabels[i], fontsize=20)
        if i == (n_params - 1):
            ax.set_xlabel("s (ps)")
    fig.savefig("coeff_vs_s_3_params_lstsq.pdf")
    fig.savefig("coeff_vs_s_3_params_lstsq.png")

    plt.figure()
    plt.plot(dt_frame*all_s, all_cv_scores)
    plt.xlabel("s (ps)")
    plt.ylabel("CV Score")
    plt.savefig("cv_score_vs_s_3_params_lstsq.pdf")
    plt.savefig("cv_score_vs_s_3_params_lstsq.png")
    plt.show()

    raise SystemExit

    q = np.diag(D_T_D)
    cond_D = np.copy(D_T_D)
    for i in range(len(cond_D)):
        cond_D[i,:] /= q 
    for j in range(len(cond_D)):
        cond_D[:,j] /= q 

    cond_Y = np.copy(D_T_Y)/q

    u_eig, sing_vals, v_eig = np.linalg.svd(G)
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







