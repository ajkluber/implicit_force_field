import time
import numpy as np

from sklearn.cross_validation import KFold

import mdtraj as md

def get_n_frames(trajfile, topfile):
    n_frames_tot = 0
    for chunk in md.iterload(trajfile, top=topfile):
        n_frames_tot += chunk.n_frames
    n_frames_tot = float(n_frames_tot)
    n_dim = 3*chunk.xyz.shape[1]
    return n_frames_tot, n_dim


def calc_deriv_and_drift(trajfile, topfile, dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, s_frames, s, n_dim, n_frames_tot):
    n_params = len(dU_funcs)
    
    G = np.zeros((int(n_frames_tot)*n_dim, n_params), float)
    Y = np.zeros(int(n_frames_tot)*n_dim, float)

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

    G = G[:start_idx]
    Y = Y[:start_idx]
    return G, Y

def solve_coefficients(trajfile, topfile, dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, s_frames, s, n_folds=10, n_blocks=False):
    print "calculating trajectory derivatives..."
    starttime = time.time()

    n_params = len(dU_funcs)
    n_frames_tot, n_dim = get_n_frames(trajfile, topfile)

    if n_blocks:
        chunksize = int(n_frames_tot)/n_blocks
        total_n_iters = int(np.round(n_frames_tot/chunksize))
        iteration_idx = 0
        c_solns = []
        all_cv_scores = []
        for chunk in md.iterload(trajfile, top=topfile, chunk=chunksize):
            # solve the problem on each chunk
            G = np.zeros(((chunk.n_frames - s_frames)*n_dim, n_params), float)

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
                xi_ravel_idxs = np.arange(dU_dxi[i], ravel_size, n_dim)
                G[xi_ravel_idxs, dU_ck[i]] += deriv.ravel()

            # calculate drift
            Y = ((xyz_flat[s_frames:,:] - xyz_flat[:-s_frames,:])/s).ravel()

            c_chunk = np.linalg.lstsq(G, Y)[0]
            c_solns.append(c_chunk)

            #chunk_cv_score = 0
            #kf = KFold(Y.shape[0], n_folds=3, shuffle=True)
            #for train_idxs, test_idxs in kf:
            #    # cross-validation: solve regression on one part of data then test
            #    # it on another. Helps measure predictability.
            #    c_new = np.linalg.lstsq(G[train_idxs], Y[train_idxs])[0]
            #    y_fit_new = np.dot(G, c_new)
            #    chunk_cv_score += np.linalg.norm(Y[test_idxs] - y_fit_new[test_idxs], ord=2)
            #chunk_cv_score /= float(3)
            #all_cv_scores.append(chunk_cv_score)

            iteration_idx += 1
        #cv_score = np.mean(all_cv_scores)
        cv_score = 0

    else:
        # calculate deriviative matrix on all data
        G, Y = calc_deriv_and_drift(trajfile, topfile, dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, s_frames, s, n_dim, n_frames_tot)
        
        cv_score = 0
        c_solns = []
        kf = KFold(Y.shape[0], n_folds=n_folds, shuffle=True)
        for train_idxs, test_idxs in kf:
            # cross-validation: solve regression on one part of data then test
            # it on another. Helps measure predictability.
            c_new = np.linalg.lstsq(G[train_idxs], Y[train_idxs])[0]
            y_fit_new = np.dot(G, c_new)
            cv_score += np.linalg.norm(Y[test_idxs] - y_fit_new[test_idxs], ord=2)
            c_solns.append(c_new)
        cv_score /= float(n_folds)

    stoptime = time.time()
    runmin = (stoptime - starttime)/60.
    print "calculation took: {} min".format(runmin)

    return c_solns, cv_score 
