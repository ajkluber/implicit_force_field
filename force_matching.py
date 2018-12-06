import numpy as np

import mdtraj as md

import basis_library

if __name__ == "__main__":
    n_beads = 25 
    n_dim = 3*n_beads
    n_folds = 10
    dt_frame = 0.2

    #topfile = "c25_min_ply_1.pdb"
    #trajfile = "c25_traj_ply_1.dcd"
    topfile = "c25_min_1.pdb"
    trajfile = "c25_traj_1.dcd"
    forcefile = "c25_forces_1.dat"

    # TODO: Update script with object oriented implementation
    print "building basis function database..."
    dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, scale_factors = basis_library.polymer_library(n_beads, gaussians=True)
    n_basis_deriv = len(dU_dxi)
    n_params = len(dU_funcs)

    n_frames_tot = 0
    for chunk in md.iterload(trajfile, top=topfile):
        n_frames_tot += chunk.n_frames
    n_frames_tot = float(n_frames_tot)

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
    forces_sim = np.loadtxt(forcefile).ravel()
    c_soln, residuals, rank, sing_vals = np.linalg.lstsq(G, forces_sim)
    c_soln *= scale_factors

    c_sim = np.array([kb_kj, ka_kj, eps_ply_kj])

    percent_error = 100*(c_soln - c_sim)/c_sim

    print "Parm    True         Soln       % Err"
    print "kb    {:10.2f}  {:10.2f}  {:10.8f}".format(c_sim[0], c_soln[0], percent_error[0])
    print "ka    {:10.2f}  {:10.2f}  {:10.8f}".format(c_sim[1], c_soln[1], percent_error[1])
    print "eps   {:10.2f}  {:10.2f}  {:10.8f}".format(c_sim[2], c_soln[2], percent_error[2])
