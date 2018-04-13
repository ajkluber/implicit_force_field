import time
import sys
import numpy as np
import scipy.linalg as scl

from sklearn.cross_validation import KFold


import mdtraj as md

def get_n_frames(trajfile, topfile):
    n_frames_tot = 0
    for chunk in md.iterload(trajfile, top=topfile):
        n_frames_tot += chunk.n_frames
    n_frames_tot = float(n_frames_tot)
    n_dim = 3*chunk.xyz.shape[1]
    return n_frames_tot, n_dim

def calc_derivative(xyz_flat, s, s_frames, n_dim, n_params, n_rows, dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, G=None):

    if G is None:
        G = np.zeros((n_rows, n_params), float)

    # calculate forces
    for i in range(len(dU_dxi)):
        # derivative 
        deriv_fun = dU_funcs[dU_ck[i]][dU_d_arg[i]]
        deriv = deriv_fun(*xyz_flat[:,dU_idxs[i]].T)[:-s_frames]   # derivative k dxi_idx = dU_dxi[i]

        # unraveled indices for xi 
        xi_ravel_idxs = np.arange(dU_dxi[i], n_rows, n_dim)
        G[xi_ravel_idxs, dU_ck[i]] += deriv.ravel()
    return G


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
            sys.stdout.flush()
        xyz_flat = np.reshape(chunk.xyz, (chunk.n_frames, n_dim))

        n_rows = (chunk.n_frames - s_frames)*n_dim
        G = calc_derivative(xyz_flat, s, s_frames, n_dim, n_params, n_rows, dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, G=G)

        # calculate drift
        Y_il = (xyz_flat[s_frames:,:] - xyz_flat[:-s_frames,:])/s
        Y[start_idx:start_idx + n_rows] = Y_il.ravel()

        iteration_idx += 1
        start_idx += n_rows

    G = G[:start_idx]
    Y = Y[:start_idx]
    return G, Y

def solve_coefficients(trajfile, topfile, dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, s_frames, s, n_folds=10, method="full", n_chunks=50):
    print "calculating trajectory derivatives..."
    starttime = time.time()

    n_params = len(dU_funcs)
    n_frames_tot, n_dim = get_n_frames(trajfile, topfile)

    # solution methods:
    #   - Full regression problem.
    #   - Chunked regression problem.
    #   - QR factorization

    if method == "full":
        # calculate deriviative matrix on all data
        G, Y = calc_deriv_and_drift(trajfile, topfile, dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, s_frames, s, n_dim, n_frames_tot)
        
        c_soln = np.linalg.lstsq(G, Y)[0]
        cv_score = 0
        c_solns = [c_soln]
        #c_solns = []
        #kf = KFold(Y.shape[0], n_folds=n_folds, shuffle=True)
        #for train_idxs, test_idxs in kf:
        #    # cross-validation: solve regression on one part of data then test
        #    # it on another. Helps measure predictability.
        #    c_new = np.linalg.lstsq(G[train_idxs], Y[train_idxs])[0]
        #    y_fit_new = np.dot(G, c_new)
        #    cv_score += np.linalg.norm(Y[test_idxs] - y_fit_new[test_idxs], ord=2)
        #    c_solns.append(c_new)
        #cv_score /= float(n_folds)
    elif method == "chunks":
        # calculate matrix on chunk of data

        chunksize = int(n_frames_tot)/n_chunks
        total_n_iters = int(np.round(n_frames_tot/chunksize))
        iteration_idx = 0
        c_solns = []
        all_cv_scores = []
        for chunk in md.iterload(trajfile, top=topfile, chunk=chunksize):
            # solve the problem on each chunk
            G = np.zeros(((chunk.n_frames - s_frames)*n_dim, n_params), float)

            if ((iteration_idx + 1) % 10) == 0:
                print "  ({}/{})".format(iteration_idx + 1, total_n_iters)
                sys.stdout.flush()
            xyz_flat = np.reshape(chunk.xyz, (chunk.n_frames, n_dim))

            n_rows = (chunk.n_frames - s_frames)*n_dim

            # calculate forces
            for i in range(len(dU_dxi)):
                # derivative 
                deriv_fun = dU_funcs[dU_ck[i]][dU_d_arg[i]]
                deriv = deriv_fun(*xyz_flat[:,dU_idxs[i]].T)[:-s_frames]   # derivative k dxi_idx = dU_dxi[i]

                # unraveled indices for xi 
                xi_ravel_idxs = np.arange(dU_dxi[i], n_rows, n_dim)
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
    elif method == "qr":
        # calculate qr factorization on all data

        #chunksize = 1000
        chunksize = int(n_frames_tot)/n_chunks
        total_n_iters = int(np.round(n_frames_tot/chunksize))
        iteration_idx = 0
        max_rows = (chunksize - s_frames)*n_dim

        for chunk in md.iterload(trajfile, top=topfile, chunk=chunksize):
            # solve the problem on each chunk
            if ((iteration_idx + 1) % 10) == 0:
                print "  ({}/{})".format(iteration_idx + 1, total_n_iters)
                sys.stdout.flush()

            n_rows = (chunk.n_frames - s_frames)*n_dim
            xyz_flat = np.reshape(chunk.xyz, (chunk.n_frames, n_dim))
            G = calc_derivative(xyz_flat, s, s_frames, n_dim, n_params, n_rows, dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck)

            # calculate drift
            Y = ((xyz_flat[s_frames:,:] - xyz_flat[:-s_frames,:])/s).ravel()

            if iteration_idx == 0:
                Q, R = scl.qr(G, mode="economic")

                X = np.zeros((n_params + max_rows, n_params), float)
                b = np.zeros(n_params + max_rows, float)

                X[:n_params,:] = R.copy()
                b[:n_params] = np.dot(Q.T, Y)
            else:
                X[n_params:,:] = G
                b[n_params:] = Y

                Qk, Rk = scl.qr(X[:n_rows,:], mode="economic")

                X[:n_params,:] = Rk
                b[:n_params] = np.dot(Qk.T, b[:n_rows])
                
            iteration_idx += 1
        final_R = X[:n_params,:]
        final_b = b[:n_params]
        c_soln = np.linalg.lstsq(final_R, final_b)[0]
        cv_score = 0
        c_solns = [c_soln]

    stoptime = time.time()
    runmin = (stoptime - starttime)/60.
    print "calculation took: {} min".format(runmin)
    sys.stdout.flush()

    return c_solns, cv_score 

def calc_diffusion(trajfile, topfile, beta, s_frames, s, n_dim, n_frames_tot):
    A = np.zeros((n_dim, n_dim), float)

    avg_dxi_dxj = np.zeros((n_dim, n_dim), float) 
    avg_dxi = np.zeros(n_dim, float)

    total_n_iters = int(np.round(n_frames_tot/1000))
    iteration_idx = 0
    N = 0
    for chunk in md.iterload(trajfile, top=topfile, chunk=1000):
        if ((iteration_idx + 1) % 10) == 0:
            print "  ({}/{})".format(iteration_idx + 1, total_n_iters)
            sys.stdout.flush()

        xyz_flat = np.reshape(chunk.xyz, (chunk.n_frames, n_dim))
        dx = xyz_flat[s_frames:] - xyz_flat[:-s_frames]
        avg_dxi_dxj += np.dot(dx.T, dx)
        avg_dxi += dx
        N += chunk.n_frames - s_frames
    
    avg_dxi_avg_dxj = np.outer(avg_dxi, avg_dxi)
    D = (beta/(2*s*float(N)))*avg_dxi_dxj
    D_stock = (beta/(2*s*float(N)))*(avg_dxi_dxj - avg_dxi_avg_dxj)

    return D, D_stock
