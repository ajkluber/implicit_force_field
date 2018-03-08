import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold

import mdtraj as md

import basis_library
import util

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
    dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, scale_factors = basis_library.polymer_library(n_beads, gaussians=True)
    n_basis_deriv = len(dU_dxi)
    n_params = len(dU_funcs)

    raise SystemExit

    all_s = [1, 2, 5, 10, 20, 50, 100, 500]
    all_cv_scores = [] 
    all_c_soln = []
    for z in range(len(all_s)):
        #for z in [0]:
        s_frames = all_s[z]
        s = dt_frame*s_frames

        print "calculating trajectory derivatives..."
        G, Y = util.calc_deriv_and_drift(trajfile, topfile, dU_funcs, dU_ck, dU_d_arg, dU_idxs, n_dim, n_params)

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







