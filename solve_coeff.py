import os
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('trajfile', help='Trajectory file.')
    parser.add_argument('topfile', help='Trajectory file.')
    parser.add_argument('--dt_frame', default=0.2, type=float, help='Timestep of one frame.')
    parser.add_argument('--in_blocks', action="store_true", help='Trajectory file.')
    parser.add_argument('--non_bond_gaussians', action="store_true", help='Trajectory file.')
    parser.add_argument('--non_bond_wca', action="store_true", help='Trajectory file.')
    parser.add_argument('--bonds', action="store_true", help='Trajectory file.')
    parser.add_argument('--angles', action="store_true", help='Trajectory file.')

    args = parser.parse_args()
    trajfile = args.trajfile
    topfile = args.topfile
    in_blocks = args.in_blocks
    dt_frame = args.dt_frame
    bonds = args.bonds
    angles = args.angles
    non_bond_wca = args.non_bond_wca
    non_bond_gaussians = args.non_bond_gaussians

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import mdtraj as md

    import basis_library
    import util

    n_beads = 25 
    n_dim = 3*n_beads
    n_folds = 10
    #dt_frame = 0.2

    #topfile = "c25_min_ply_1.pdb"
    #trajfile = "c25_traj_ply_1.dcd"
    #topfile = "c25_min_1.pdb"
    #trajfile = "c25_traj_1.dcd"
    #topfile = "c25_min_ply_1.pdb"
    #trajfile = "c25_traj_ply_1.dcd"

    if in_blocks:
        n_blocks = 50
    else:
        n_blocks = False

    print "building basis function database..."
    dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, scale_factors = basis_library.polymer_library(n_beads, bonds=bonds, angles=angles, non_bond_wca=non_bond_wca, non_bond_gaussians=non_bond_gaussians)
    n_basis_deriv = len(dU_dxi)
    n_params = len(dU_funcs)

    all_s = [1, 2, 5, 10, 20, 50, 100, 500]
    all_cv_scores = np.zeros(len(all_s), float)
    all_c_soln = []
    for z in range(len(all_s)):
        #for z in [0]:
        s_frames = all_s[z]
        s = dt_frame*s_frames
        c_solns, cv_score = util.solve_coefficients(trajfile, topfile, dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, s_frames, s, n_folds=10, n_blocks=n_blocks)
        all_c_soln.append(c_solns)
        all_cv_scores[z] = cv_score
    all_c_soln = np.array(all_c_soln)
    all_s = np.array(all_s)

    c_vs_s = []
    for n in range(n_params):
        if n in [0,1,2]:
            c_vs_s.append(scale_factors[n]*np.array([ np.array(all_c_soln[i])[:,n] for i in range(all_c_soln.shape[0]) ]))
        else:
            c_vs_s.append(scale_factors[-1]*np.array([ np.array(all_c_soln[i])[:,n] for i in range(all_c_soln.shape[0]) ]))

    raise SystemExit
    avg_c_vs_s = [] 
    for i in range(10):
        coeff = c_vs_s[i + 3]
        avg_c_vs_s.append(np.mean(coeff, axis=1))

    r = np.linspace(0.2, 1.5, 1000)
    y = np.zeros(len(r), float)
    for i in range(len(avg_c_vs_s)):
        c_k = avg_c_vs_s[i][1]
        y += c_k*gauss_funcs[i](r)


    plt.figure()
    for i in range(10):
        coeff = c_vs_s[i + 3]
        plt.errorbar(dt_frame*all_s[:len(coeff)], np.mean(coeff, axis=1), yerr=np.std(coeff, axis=1))
    plt.ylabel(r"$c_k$", fontsize=20)
    plt.savefig("coeff_gauss_vs_s.pdf")
    plt.savefig("coeff_gauss_vs_s.png")

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







