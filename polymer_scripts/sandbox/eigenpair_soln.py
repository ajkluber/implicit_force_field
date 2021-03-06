from __future__ import print_function
import os
import glob
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import sklearn.linear_model as sklin

import mdtraj as md

import simulation.openmm as sop

import implicit_force_field as iff
import implicit_force_field.polymer_scripts.util as util

def plot_train_test_mse(alphas, train_mse, test_mse, 
        xlabel=r"Regularization $\alpha$", ylabel="Mean squared error (MSE)", 
        title="", prefix=""):
    """Plot mean squared error for training and test data"""

    alpha_star = alphas[np.argmin(test_mse[:,0])]

    plt.figure()
    ln1 = plt.plot(alphas, train_mse[:,0], label="Training set")[0]
    ln2 = plt.plot(alphas, test_mse[:,0], label="Test set")[0]
    plt.fill_between(alphas, train_mse[:,0] + train_mse[:,1],
            y2=train_mse[:,0] - train_mse[:,1],
            facecolor=ln1.get_color(), alpha=0.5)

    plt.fill_between(alphas, test_mse[:,0] + test_mse[:,1],
            y2=test_mse[:,0] - test_mse[:,1],
            facecolor=ln2.get_color(), alpha=0.5)

    plt.axvline(alpha_star, color='k', ls='--', label=r"$\alpha^* = {:.2e}$".format(alpha_star))
    plt.semilogx(True)
    plt.semilogy(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(prefix + "train_test_mse.pdf")
    plt.savefig(prefix + "train_test_mse.png")

def plot_Ucg_vs_psi1(coeff, Ucg, cv_r0, prefix):

    U = np.zeros(len(cv_r0))
    for i in range(len(coeff) - 1):
        U += coeff[i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
    U -= U.min()

    plt.figure()
    plt.plot(cv_r0[:,0], U)
    plt.xlabel(r"TIC1 $\psi_1$")
    plt.ylabel(r"$U_{cg}(\psi_1)$")
    plt.savefig("{}U_cv.pdf".format(prefix))
    plt.savefig("{}U_cv.png".format(prefix))

def plot_Ucg_vs_alpha(idxs, idx_star, coeffs, alphas, Ucg, cv_r0, prefix, ylim=None):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for n in range(len(idxs)):

        coeff = coeffs[idxs[n]]

        U = np.zeros(len(cv_r0))
        for i in range(len(coeff) - 1):
            U += coeff[i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
        U -= U.min()

        D = 1./coeff[-1]
        ax1.plot(cv_r0[:,0], U, label=r"$\alpha={:.2e}$".format(alphas[idxs[n]]))
        ax2.plot(cv_r0[:,0], D*np.ones(len(cv_r0[:,0])))

    coeff = coeffs[idx_star]
    U = np.zeros(len(cv_r0))
    for i in range(len(coeff) - 1):
        U += coeff[i]*Ucg.cv_U_funcs[i](cv_r0[:,0])
    U -= U.min()
    D = 1./coeff[-1]
    ax1.plot(cv_r0[:,0], U, color='k', lw=3, label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))
    ax2.plot(cv_r0[:,0], D*np.ones(len(cv_r0[:,0])), color='k', lw=3)
    ax2.semilogy(True)

    if not ylim is None:
        #ax1.set_ylim(0, 500)
        ax1.set_ylim(0, ylim)

    ax1.legend()
    ax1.set_xlabel(r"TIC1 $\psi_1$")
    ax1.set_ylabel(r"$U_{cg}(\psi_1)$")
    ax2.set_xlabel(r"TIC1 $\psi_1$")
    ax2.set_ylabel(r"$D$")
    fig.savefig("{}compare_Ucv.pdf".format(prefix))
    fig.savefig("{}compare_Ucv.png".format(prefix))

def plot_Xcoeff_vs_d(idxs, idx_star, coeffs, alphas, X, d, prefix):

    plt.figure()
    for i in range(len(idxs)):
        dfit = np.dot(X, coeffs[idxs[i]])
        plt.plot((dfit - d)**2, '.', label=r"$\alpha={:.2e}$".format(alphas[idxs[i]]))

    dfit = np.dot(X, coeffs[idx_star])
    plt.plot((dfit - d)**2, 'k.', label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))
    plt.ylabel(r"$(\mathbf{X}c^* - \mathbf{d})^2$")
    plt.legend()
    plt.savefig("{}compare_d_residual.pdf".format(prefix))
    plt.savefig("{}compare_d_residual.png".format(prefix))

    plt.figure()
    plt.plot(d, 'ko', label="Target $\mathbf{d}$")
    for i in range(len(idxs)):
        dfit = np.dot(X, coeffs[idxs[i]])
        plt.plot(dfit, '.', label=r"$\alpha={:.2e}$".format(alphas[idxs[i]]))

    dfit = np.dot(X, coeffs[idx_star])
    plt.plot(dfit, '.', label=r"$\alpha^*={:.2e}$".format(alphas[idx_star]))
    plt.ylabel(r"$\mathbf{d}$  or  $\mathbf{X}c^*$ ")
    plt.legend()
    plt.savefig("{}compare_d_fit_target.pdf".format(prefix))
    plt.savefig("{}compare_d_fit_target.png".format(prefix))

def get_frame_assignments(trajnames, n_cross_val_sets=5):
    """Randomly assign trajectory frames to sets"""
    set_assignment = []
    traj_n_frames = []
    for n in range(len(trajnames)):
        length = 0
        for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
            length += chunk.n_frames
        traj_n_frames.append(length)

        set_assignment.append(np.random.randint(low=0, high=n_cross_val_sets, size=length))
    total_n_frames = sum(traj_n_frames)

    return set_assignment

def split_trajs_into_train_and_test_sets(trajnames, psinames, n_cross_val_sets=5):
    """Split trajs into sets with roughly same number of frames

    Trajectories are assigned into sets which will be used as training and test
    data sets. For simplicity they are assigned whole, so the number of frames
    in each set will vary.
    
    Parameters
    ----------
    trajnames : list, str
        List of trajectory filenames.

    total_n_frames : int, opt.
        Total number of trajectory frames.

    n_cross_val_sets : int, default=5
        Desired number of trajectory sets.
        
    """

    traj_n_frames = []
    for n in range(len(trajnames)):
        length = 0
        for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
            length += chunk.n_frames
        traj_n_frames.append(length)
    total_n_frames = sum(traj_n_frames)

    n_frames_in_set = total_n_frames/n_cross_val_sets

    traj_set = []
    psi_set = []
    traj_set_frames = []
    temp_traj_names = []
    temp_psi_names = []
    temp_frame_count = 0
    for n in range(len(traj_n_frames)):
        if temp_frame_count >= n_frames_in_set:
            # finish a set when it has desired number of frames
            traj_set.append(temp_traj_names)
            psi_set.append(temp_psi_names)
            traj_set_frames.append(temp_frame_count)

            # start over
            temp_traj_names = [trajnames[n]]
            temp_psi_names = [psinames[n]]
            temp_frame_count = traj_n_frames[n]
        else:
            temp_traj_names.append(trajnames[n])
            temp_psi_names.append(psinames[n])
            temp_frame_count += traj_n_frames[n]

        if n == len(traj_n_frames) - 1:
            traj_set.append(temp_traj_names)
            psi_set.append(temp_psi_names)
            traj_set_frames.append(temp_frame_count)

    with open("traj_sets_{}.txt".format(n_cross_val_sets), "w") as fout:
        for i in range(len(traj_set)):
            info_str = str(traj_set_frames[i])
            info_str += " " + " ".join(traj_set[i]) + "\n"
            fout.write(info_str)

    return traj_set, traj_set_frames, psi_set

if __name__ == "__main__":
    n_beads = 25
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)
    n_pair_gauss = 10
    M = 1   # number of eigenvectors to use
    #fixed_bonded_terms = False
    fixed_bonded_terms = True

    using_cv = True
    using_cv_r0 = False
    n_cv_basis_funcs = 40
    n_cv_test_funcs = 100
    #n_cv_test_funcs = 40
    #n_cv_basis_funcs = 100
    #n_cv_test_funcs = 200

    using_D2 = False
    n_cross_val_sets = 5

    #msm_savedir = "msm_dih_dists"
    msm_savedir = "msm_dists"

    Ucg, cg_savedir, cv_r0_basis, cv_r0_test = util.create_polymer_Ucg(msm_savedir, n_beads, M, beta, fixed_bonded_terms, using_cv, using_cv_r0, using_D2, n_cv_basis_funcs, n_cv_test_funcs)

    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd") 
    ti_file = msm_savedir + "/tica_ti.npy"
    psinames = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        temp_names = []
        for n in range(M):
            temp_names.append(msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, n+1))
        psinames.append(temp_names)

    cg_savedir = cg_savedir + "_crossval_{}".format(n_cross_val_sets)
    if not os.path.exists(cg_savedir):
        os.mkdir(cg_savedir)

    ##################################################################
    # calculate matrix X and d 
    ##################################################################
    X_sets_exist = [ os.path.exists("{}/X_{}.npy".format(cg_savedir, i + 1)) for i in range(n_cross_val_sets) ]
    set_assignment_exist = [ os.path.exists("{}/frame_set_{}.npy".format(cg_savedir, i + 1)) for i in range(n_cross_val_sets) ]

    if not np.all(X_sets_exist) or not np.all(set_assignment_exist):
        # Calculate matrices over randomized assigned groups of frames
        set_assignment = get_frame_assignments(trajnames, n_cross_val_sets=n_cross_val_sets)
        Ucg.setup_eigenpair(trajnames, topfile, psinames, ti_file, M=M, cv_names=psinames, set_assignment=set_assignment, verbose=True)

        X_sets = [ Ucg.eigenpair_X[k] for k in range(n_cross_val_sets) ]
        d_sets = [ Ucg.eigenpair_d[k] for k in range(n_cross_val_sets) ]
        for k in range(n_cross_val_sets):
            np.save("{}/X_{}.npy".format(cg_savedir, k + 1), Ucg.eigenpair_X[k])
            np.save("{}/d_{}.npy".format(cg_savedir, k + 1), Ucg.eigenpair_d[k])
            np.save("{}/frame_set_{}.npy".format(cg_savedir,  k + 1), set_assignment[k])
    else:
        X_sets = [ np.load("{}/X_{}.npy".format(cg_savedir, i + 1)) for i in range(n_cross_val_sets) ]
        d_sets = [ np.load("{}/d_{}.npy".format(cg_savedir, i + 1)) for i in range(n_cross_val_sets) ]
        set_assignment = [ np.load("{}/frame_set_{}.npy".format(cg_savedir, i + 1)) for i in range(n_cross_val_sets) ]

    n_frames_in_set = []
    for k in range(n_cross_val_sets):
        n_frames_in_set.append(np.sum([ np.sum(set_assignment[i] == k) for i in range(len(set_assignment)) ]))
    total_n_frames = np.sum(n_frames_in_set) 

    X = np.sum([ (n_frames_in_set[j]/float(total_n_frames))*X_sets[j] for j in range(n_cross_val_sets) ], axis=0)
    d = np.sum([ (n_frames_in_set[j]/float(total_n_frames))*d_sets[j] for j in range(n_cross_val_sets) ], axis=0)

    np.save("{}/X.npy".format(cg_savedir), X)
    np.save("{}/d.npy".format(cg_savedir), d)

    # create train/test matrices by weighted average of chunk matrices
    X_train_test = []
    d_train_test = []
    for i in range(n_cross_val_sets):
        frame_subtotal = total_n_frames - n_frames_in_set[i]
        #frame_subtotal = np.sum([ n_frames_in_set[j] for j in range(n_cross_val_sets) if j != i ])

        train_X = []
        train_d = []
        for j in range(n_cross_val_sets):
            w_j = n_frames_in_set[j]/float(frame_subtotal)
            if j != i:
                train_X.append(w_j*X_sets[j])
                train_d.append(w_j*d_sets[j])

        train_X = np.sum(np.array(train_X), axis=0)
        train_d = np.sum(np.array(train_d), axis=0)

        X_train_test.append([ train_X, X_sets[i]])
        d_train_test.append([ train_d, d_sets[i]])

    os.chdir(cg_savedir)

    print("Ridge regularization...")
    rdg_alphas = np.logspace(-10, 8, 500)
    rdg_coeffs, rdg_train_mse, rdg_test_mse = iff.util.traj_chunk_cross_validated_least_squares(rdg_alphas, X, d,
            X_train_test, d_train_test, np.identity(X.shape[1]))

    rdg_idx_star = np.argmin(rdg_test_mse[:,0])
    rdg_alpha_star = rdg_alphas[rdg_idx_star]
    rdg_cstar = rdg_coeffs[rdg_idx_star]

    # save solutions
    np.save("rdg_cstar.npy", rdg_cstar)
    with open("rdg_alpha_star.dat", "w") as fout:
        fout.write(str(rdg_alpha_star))

    print("Plotting ridge...")
    plot_train_test_mse(rdg_alphas, rdg_train_mse, rdg_test_mse, 
            xlabel=r"Regularization $\alpha$", 
            ylabel="Mean squared error (MSE)", 
            title="Ridge regression", prefix="ridge_")

    rdg_idxs = [5, 50, 200, 300]
    plot_Ucg_vs_alpha(rdg_idxs, rdg_idx_star, rdg_coeffs, rdg_alphas, Ucg, cv_r0_basis, "rdg_", ylim=200)
    plot_Xcoeff_vs_d(rdg_idxs, rdg_idx_star, rdg_coeffs, rdg_alphas, X, d, "rdg_")

    rdg_idxs = [310, 400, 472]
    plot_Ucg_vs_alpha(rdg_idxs, rdg_idx_star, rdg_coeffs, rdg_alphas, Ucg, cv_r0_basis, "rdg_2_", ylim=90)

    #plot_Ucg_vs_psi1(rdg_cstar, Ucg, cv_r0, "rdg_")

    raise SystemExit

    # Smoothness penalty only valid when using same centers as the 1D calculation
    d2_alphas = np.logspace(-10, 8, 500)
    D2 = np.zeros((101,101), float)
    D2[:100,:100] = np.load("../Ucg_eigenpair_1D/D2.npy")[:100,:100]

    print("D2 regularization...")
    #d2_coeffs, d2_train_mse, d2_test_mse = iff.util.traj_chunk_cross_validated_least_squares(d2_alphas, X, d,
    #        X_train_test, d_train_test, D2) 
    d2_coeffs, d2_train_mse, d2_test_mse = traj_chunk_cross_validated_least_squares(d2_alphas, X, d,
            X_train_test, d_train_test, D2) 


    print("Plotting D2...")
    plot_train_test_mse(d2_alphas, d2_train_mse, d2_test_mse, 
            xlabel=r"Regularization $\alpha$", 
            ylabel="Mean squared error (MSE)", 
            title="Second deriv penalty", prefix="D2_")

    d2_idx_star = np.argmin(d2_test_mse[:,0])
    d2_alpha_star = d2_alphas[d2_idx_star]
    d2_cstar = d2_coeffs[d2_idx_star]

    d2_idxs = [50, 100, 300, 480]
    plot_Ucg_vs_alpha(d2_idxs, d2_idx_star, d2_coeffs, d2_alphas, Ucg, cv_r0_basis, "D2_")

    #plot_Ucg_vs_psi1(d2_cstar, Ucg, cv_r0_basis, "D2_")

    raise SystemExit
    # Plot matrix columns
    
    fig, axes = plt.subplots(10, 10, figsize=(50,50))
    for i in range(10):
        for j in range(10):
            idx = i*10 + j
            ax = axes[i,j]
            ax.plot(cv_r0_basis[:,0], X[:,idx], 'k')
            ax.plot(cv_r0_basis[:,0], X[:,idx], 'k.')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    #fig.savefig(cg_savedir + "/Xcols.pdf")
    #fig.savefig(cg_savedir + "/Xcols.png")
    fig.savefig("Xcols.pdf")
    fig.savefig("Xcols.png")

    fig, axes = plt.subplots(5, 8, figsize=(25,40))
    for i in range(5):
        for j in range(8):
            idx = i*8 + j
            ax = axes[i,j]
            ax.plot(cv_r0_basis[:,0], X[:,idx], 'k')
            ax.plot(cv_r0_basis[:,0], X[:,idx], 'k.')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    #fig.savefig(cg_savedir + "/Xcols.pdf")
    #fig.savefig(cg_savedir + "/Xcols.png")
    fig.savefig("Xcols.pdf")
    fig.savefig("Xcols.png")

    plt.figure()
    plt.plot(cv_r0_basis[:,0], X[:,-1], 'k')
    plt.plot(cv_r0_basis[:,0], X[:,-1], 'k.')
    #plt.savefig(cg_savedir + "/Xcols_D.pdf")
    #plt.savefig(cg_savedir + "/Xcols_D.png")
    plt.savefig("Xcols_D.pdf")
    plt.savefig("Xcols_D.png")

    A_reg = np.dot(X.T, X) + rdg_alpha_star*np.identity(X.shape[1])
    b_reg = np.dot(X.T, d)
    coeff = np.linalg.lstsq(A_reg, b_reg, rcond=1e-10)[0] 
    plt.figure()
    plt.plot(np.arange(len(d)), d, 'ko', label="Target")
    plt.plot(np.arange(len(d)), np.dot(X, coeff), 'r.', label="Soln")
    plt.legend()
    plt.ylabel(r"$d$")
    plt.savefig("target_d_and_soln_d.pdf")
    plt.savefig("target_d_and_soln_d.png")

    raise SystemExit

    print("Calculating Laplacian...")
    bin_width = np.abs(cv_r0[1,0] - cv_r0[0,0])
    psi1_bin_edges = np.array(list(cv_r0[:,0] - 0.5*bin_width) + [cv_r0[-1,0] + 0.5*bin_width])
    Ucg._eigenpair_Lap_f(trajnames, topfile, psinames, psi1_bin_edges, ti_file, M=M, cv_names=psinames, verbose=True)

    avgLapf = Ucg.eigenpair_Lapf_vs_psi[0]

    fig, axes = plt.subplots(10, 10, figsize=(50,50))
    for i in range(10):
        for j in range(10):
            idx = i*10 + j
            ax = axes[i,j]
            ax.plot(cv_r0[:,0], avgLapf[idx], 'k')
            ax.plot(cv_r0[:,0], avgLapf[idx], 'k.')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    #fig.savefig(cg_savedir + "/avg_Lapf.pdf")
    #fig.savefig(cg_savedir + "/avg_Lapf.png")
    fig.savefig(cg_savedir + "/avg_Lapf.pdf")
    fig.savefig(cg_savedir + "/avg_Lapf.png")



    #P_psi = bin_width*np.load(msm_savedir + "/psi1_n.npy").astype(float)
    #P_psi /= np.sum(P_psi)
    #d2P_psi = (1/bin_width**2)*np.diff(P_psi, n=2)

    #plt.figure()
    #plt.plot(cv_r0[:,0], -beta*d, 'ko', label=r"$\langle\psi_1, \Delta_x f_j\rangle$")
    #plt.plot(cv_r0[2:,0], cv_r0[2:,0]*d2P_psi, 'r.', label=r"$P''(\psi_1)$")
    #plt.legend()
    #plt.xlabel(r"TIC 1 $\psi_1$")
    #plt.savefig(cg_savedir + "/derviv2_prob_psi.pdf")
    #plt.savefig(cg_savedir + "/derviv2_prob_psi.png")

    raise SystemExit

    temp_psi = []
    for chunk in md.iterload(trajnames[0], top=topfile):
        temp_psi.append(Ucg.calculate_cv(chunk.xyz.reshape(-1, 75))[:,0])
    temp_psi = np.concatenate(temp_psi)
    #[ x[:,0] for x in temp_psi ])

        
