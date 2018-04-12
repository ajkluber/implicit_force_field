import os
import sys
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('trajfile', help='Trajectory file.')
    parser.add_argument('topfile', help='Trajectory file.')
    parser.add_argument('--dt_frame', default=0.2, type=float, help='Timestep of one frame.')
    parser.add_argument('--gamma', default=100, type=float, help='Friction coefficient.')
    parser.add_argument('--method', type=str, default="full", help='Calculation method.')
    parser.add_argument('--non_bond_gaussians', action="store_true")
    parser.add_argument('--non_bond_wca', action="store_true")
    parser.add_argument('--bonds', action="store_true")
    parser.add_argument('--angles', action="store_true")

    args = parser.parse_args()
    trajfile = args.trajfile
    topfile = args.topfile
    method = args.method
    dt_frame = args.dt_frame
    bonds = args.bonds
    angles = args.angles
    non_bond_wca = args.non_bond_wca
    non_bond_gaussians = args.non_bond_gaussians

    #python ~/code/implicit_force_field/solve_coeff.py c25_traj_1.dcd c25_min_1.pdb --dt_frame 0.0002 --method qr --bonds --angles --non_bond_gaussians
    #python ~/code/implicit_force_field/solve_coeff.py c25_traj_1.dcd c25_min_1.pdb --dt_frame 0.0002 --method qr --bonds --angles --non_bond_wca

    assert method in ["full", "chunks", "qr"], "IOError. method must be full, chunks, or qr"
    print "method: ", method

    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    import matplotlib.pyplot as plt

    import mdtraj as md

    import basis_library
    import util

    savedir = "coeff_vs_s"
    if bonds:
        savedir += "_bond"
    if angles:
        savedir += "_angle"
    if non_bond_wca:
        savedir += "_wca"
    if non_bond_gaussians:
        savedir += "_gauss"

    savedir += "_" + method

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    n_beads = 25 
    n_dim = 3*n_beads
    n_folds = 5
    #dt_frame = 0.2
    gamma = 100
    n_chunks = 50

    print "building basis function database..."
    sys.stdout.flush()
    dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, scale_factors = basis_library.polymer_library(n_beads, bonds=bonds, angles=angles, non_bond_wca=non_bond_wca, non_bond_gaussians=non_bond_gaussians)
    n_basis_deriv = len(dU_dxi)
    n_params = len(dU_funcs)

    all_s = [1, 2, 5, 10, 20, 50, 100, 500]
    #all_s = [1, 5, 10]
    all_c_soln = []
    all_cv_scores = []
    for z in range(len(all_s)):
        #for z in [0]:
        s_frames = all_s[z]
        s = dt_frame*s_frames
        c_solns, cv_score = util.solve_coefficients(trajfile, topfile, dU_funcs, dU_idxs, dU_d_arg, dU_dxi, dU_ck, s_frames, s, n_folds=n_folds, method=method, n_chunks=n_chunks)

        for n in range(n_params):
            if non_bond_gaussians and not (bonds and angles):
                temp_coeff = scale_factors[0]*np.array(c_solns)[:,n]
            else:
                if n in [0,1,2]:
                    temp_coeff = scale_factors[n]*np.array(c_solns)[:,n]
                else:
                    temp_coeff = scale_factors[-1]*np.array(c_solns)[:,n]
            np.save("{}/coeff_{}_s_{}.npy".format(savedir, n+1, s_frames), temp_coeff)

        all_c_soln.append(c_solns)
        all_cv_scores.append(cv_score)
    all_s = np.array(all_s)
    all_c_soln = np.array(all_c_soln)
    all_cv_scores = np.array(all_cv_scores)

    c_vs_s = []
    for n in range(n_params):
        if non_bond_gaussians and not (bonds and angles):
            c_vs_s.append(scale_factors[0]*np.array([ np.array(all_c_soln[i])[:,n] for i in range(all_c_soln.shape[0]) ]))
        else:
            if n in [0,1,2]:
                c_vs_s.append(scale_factors[n]*np.array([ np.array(all_c_soln[i])[:,n] for i in range(all_c_soln.shape[0]) ]))
            else:
                c_vs_s.append(scale_factors[-1]*np.array([ np.array(all_c_soln[i])[:,n] for i in range(all_c_soln.shape[0]) ]))

    os.chdir(savedir)
    # save coefficients versus lagtime
    for i in range(n_params):
        np.save("coeff_{}_vs_s.npy".format((i+1)), c_vs_s[i])
    np.save("s_list.npy", all_s)
    np.save("cv_score.npy", all_cv_scores)

    if non_bond_gaussians:
        avg_c_vs_s = [] 
        n_gaussians = len(c_vs_s[3:])
        for i in range(10):
            coeff = c_vs_s[i + 3]
            avg_c_vs_s.append(np.mean(coeff, axis=1))

        r = np.linspace(0.2, 1.5, 1000)
        y = np.zeros(len(r), float)
        for i in range(len(avg_c_vs_s)):
            c_k = avg_c_vs_s[i][1]
            y += c_k*gauss_funcs[i](r)
        y *= gamma

        plt.figure()
        plt.plot(r, y)
        plt.xlabel("Distance (nm)")
        plt.ylabel("Effective potential")
        plt.savefig("gaussian_interaction.pdf")
        plt.savefig("gaussian_interaction.png")

        plt.figure()
        for i in range(10):
            coeff = c_vs_s[i + 3]
            plt.errorbar(dt_frame*all_s[:len(coeff)], np.mean(coeff, axis=1), yerr=np.std(coeff, axis=1))
        plt.ylabel(r"$c_k$", fontsize=20)
        plt.savefig("coeff_gauss_vs_s.pdf")
        plt.savefig("coeff_gauss_vs_s.png")

    ylabels = [r"$k_b$", r"$k_a$", r"$\epsilon$"]
    #coeff_true = [334720, 462, 0.14] 
    coeff_true = [100, 20, 1] 
    s_complete = all_s[:len(c_vs_s[0])]


    if bonds and angles:
        fig, axes = plt.subplots(3, 1, figsize=(4,10))
        for i in range(n_params):
            coeff = c_vs_s[i]

            c_avg = gamma*np.mean(coeff, axis=1)
            c_avg_err = gamma*np.std(coeff, axis=1)/np.sqrt(float(coeff.shape[1]))
            #c_avg = np.mean(coeff, axis=1)
            #c_avg_err = np.std(coeff, axis=1)/np.sqrt(float(coeff.shape[1]))

            #print c_avg, c_avg_err

            ax = axes[i]
            ax.errorbar(dt_frame*s_complete, c_avg, yerr=c_avg_err)
            #ax.axhline(coeff_true[i], color='k', ls='--', label="True")
            ax.set_ylabel(ylabels[i], fontsize=26)
            if i == (n_params - 1):
                ax.set_xlabel(r"$\Delta t$ (ps)", fontsize=26)
            #ax.set_xlim(0, 1.1*dt_frame*np.max(s_complete))
            ax.set_xlim(0.9*dt_frame, 1.1*dt_frame*np.max(s_complete))
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, 1.2*ymax)
            ax.semilogx()
            #ax.semilogy()
        fig.savefig("coeff_vs_s_3_params_lstsq_xlog.pdf")
        fig.savefig("coeff_vs_s_3_params_lstsq_xlog.png")
        #fig.savefig("coeff_vs_s_3_params_lstsq_xlog_with_true.pdf")
        #fig.savefig("coeff_vs_s_3_params_lstsq_xlog_with_true.png")

        plt.figure()
        plt.plot(dt_frame*s_complete, all_cv_scores[:len(c_vs_s[0])])
        plt.xlabel("s (ps)")
        plt.ylabel("CV Score")
        plt.savefig("cv_score_vs_s_3_params_lstsq.pdf")
        plt.savefig("cv_score_vs_s_3_params_lstsq.png")

    raise SystemExit
    line_colors = ["#ff7f00", "#377eb8", "grey"] # orange, blue, grey

    plt.figure()
    for i in range(n_params):
        coeff = c_vs_s[i]
        c_avg = gamma*np.mean(coeff, axis=1)
        c_avg_err = gamma*np.std(coeff, axis=1)/np.sqrt(float(coeff.shape[1]))
        c_avg /= coeff_true[i]
        c_avg_err /= coeff_true[i]
        c_avg *= 40
        c_avg_err *= 40

        plt.errorbar(dt_frame*s_complete, c_avg, yerr=c_avg_err, color=line_colors[i], label=ylabels[i])

    #plt.legend(loc=1, fontsize=20, fancybox=False, frameon=True, edgecolor="k", framealpha=1)
    legend = plt.legend(loc=1, fontsize=20, fancybox=False, frameon=True)
    legend.get_frame().set_edgecolor("k") 
    plt.ylabel("Effective / True", fontsize=26)
    plt.xlim(0.9*dt_frame, 1.1*dt_frame*np.max(s_complete))
    plt.semilogx()
    plt.xlabel(r"$\Delta t$ (ps)", fontsize=26)
    plt.savefig("coeff_vs_s_3_params_lstsq_ratio_xlog.pdf")
    plt.savefig("coeff_vs_s_3_params_lstsq_ratio_xlog.png")
    #plt.savefig("coeff_vs_s_3_params_lstsq_xlog_with_true.pdf")
    #plt.savefig("coeff_vs_s_3_params_lstsq_xlog_with_true.png")

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









