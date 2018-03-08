import time
import numpy as np

import mdtraj as md

def get_n_frames(trajfile, topfile):
    n_frames_tot = 0
    for chunk in md.iterload(trajfile, top=topfile):
        n_frames_tot += chunk.n_frames
    n_frames_tot = float(n_frames_tot)
    return n_frames_tot

def calc_deriv_and_drift(trajfile, topfile, s_frames, dU_funcs, dU_ck, dU_d_arg, dU_idxs, n_dim):

    n_params = len(dU_funcs)
    n_frames_tot = get_n_frames(trajfile, topfile)
    
    G = np.zeros((int(n_frames_tot)*n_dim, n_params), float)
    Y = np.zeros(int(n_frames_tot)*n_dim, float)

    starttime = time.time()
    start_idx = 0
    total_n_iters = int(np.round(n_frames_tot/1000))
    iteration_idx = 0
    for chunk in md.iterload(trajfile, top=topfile, chunk=1000):
        if ((iteration_idx + 1) % 10) == 0:
            print "  ({}/{})".format(iteration_idx + 1, total_n_iters)
        xyz_flat = np.reshape(chunk.xyz, (chunk.n_frames, n_dim))

        ravel_size = (chunk.n_frames - s_frames)*n_dim

        # calculate forces
        for i in range(len(dU_dxi)):
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
