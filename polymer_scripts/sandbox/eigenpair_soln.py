import os
import sys
import glob
import time
import argparse
import numpy as np
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt

import simtk.unit as unit
import simtk.openmm.app as app

import mdtraj as md

import simulation.openmm as sop
import implicit_force_field as iff

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_idx", type=int)
    parser.add_argument("n_gauss", type=int)
    args = parser.parse_args()

    run_idx = args.run_idx
    n_gauss = args.n_gauss

    n_beads = 25
    #n_beads = 5
    #n_beads = 4
    #n_beads = 3
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)

    traj_idx = 1
    #topname = "c25_nosolv_min.pdb"
    topname = name + "_min.pdb"
    min_name = name + "_min_{}.pdb".format(traj_idx)
    log_name = name + "_{}.log".format(traj_idx)
    traj_name = name + "_traj_cent_{}.dcd".format(traj_idx)
    lastframe_name = name + "_fin_{}.pdb".format(traj_idx)


    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    r0, kb, theta0, ka = bonded_params
    app.element.polymer = app.element.Element(200, "Polymer", "Pl", mass_ply)

    sigma_ply_nm = sigma_ply/unit.nanometer
    #r0_wca_nm = sigma_ply_nm*(2**(1./6))
    eps_ply_kj = eps_ply/unit.kilojoule_per_mole
    kb_kj = kb/(unit.kilojoule_per_mole/(unit.nanometer**2))
    ka_kj = (ka/(unit.kilojoule_per_mole/(unit.radian**2)))
    theta0_rad = theta0/unit.radian
    r0_nm = r0/unit.nanometer

    #gauss_r0_nm = np.linspace(0.3, 1, 10)
    #gauss_w_nm = 0.1*np.ones(len(gauss_r0_nm))

    gauss_r0_nm = np.linspace(0.3, 1, n_gauss)
    gauss_sigma = gauss_r0_nm[1] - gauss_r0_nm[0]
    gauss_w_nm = gauss_sigma*np.ones(len(gauss_r0_nm))

    #n_gauss = len(gauss_r0_nm)

    #msm_savedir = "msm_dih_dists"
    msm_savedir = "msm_dists"

    #M = 3
    M = 1   # number of eigenvectors to use

    #cg_savedir = "Ucg_eigenpair_fixed_bonds_angles_free_pairs_10"
    cg_savedir = "Ucg_eigenpair_fixed_bonds_angles_free_pairs_{}_CV_{}".format(n_gauss, M)

    # scaling the terms of the potential reduces the condition number of the
    # force matrix by several orders of magnitude. Pre-conditioning by column
    # multiplication

    print "creating Ucg..."
    # coarse-grain polymer potential with free parameters
    Ucg = iff.basis_library.PolymerModel(n_beads)
    Ucg.harmonic_bond_potentials(r0_nm, scale_factor=kb_kj, fixed=True)
    Ucg.harmonic_angle_potentials(theta0_rad, scale_factor=ka_kj, fixed=True)
    #Ucg.LJ6_potentials(sigma_ply_nm, scale_factor=eps_ply_kj)
    Ucg.inverse_r12_potentials(sigma_ply_nm, scale_factor=0.5, fixed=True)
    Ucg.gaussian_pair_potentials(gauss_r0_nm, gauss_w_nm, scale_factor=10)


    ply_idxs = np.arange(25)
    pair_idxs = []
    for i in range(len(ply_idxs) - 1):
        for j in range(i + 4, len(ply_idxs)):
            pair_idxs.append([ply_idxs[i], ply_idxs[j]])
    pair_idxs = np.array(pair_idxs)

    # add test functions
    use_cv_f_j = True
    if use_cv_f_j:
        # centers of test functions in collective variable (CV) space
        cv_r0 = np.load(msm_savedir + "/psi1_mid_bin.npy")
        cv_w = np.abs(cv_r0[1] - cv_r0[0])*np.ones(len(cv_r0), float)
        #cv_r0 = np.array([ [cv_r0[i]] for i in range(len(cv_r0)) ])
        cv_r0 = cv_r0.reshape((len(cv_r0),1))

        cv_coeff = np.load(msm_savedir + "/tica_eigenvects.npy")[:,:M]
        cv_mean = np.load(msm_savedir + "/tica_mean.npy")

        # TODO: add additional features
        Ucg.linear_collective_variables(["dist"], pair_idxs, cv_coeff, cv_mean)
        Ucg.gaussian_cv_test_funcs(cv_r0, cv_w)
    else:
        Ucg.gaussian_bond_test_funcs([r0_nm], [0.3])
        Ucg.vonMises_angle_test_funcs([theta0_rad], [4])
        Ucg.gaussian_pair_test_funcs(gauss_r0_nm, gauss_w_nm)

    D = Ucg.n_dof           # number of degrees of freedom
    R = Ucg.n_params        # number of free model parameters
    if use_cv_f_j:
        P = Ucg.n_test_funcs_cv # number of test functions
    else:
        P = Ucg.n_test_funcs    # number of test functions

    ##########################################################
    # calculate integrated sindy (eigenpair) matrix equation.
    ########################################################## 
    # load tics
    topfile = glob.glob("run_{}/".format(run_idx) + name + "_min_cent.pdb")[0]
    trajnames = glob.glob("run_{}/".format(run_idx) + name + "_traj_cent_*.dcd") 
    traj_idxs = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        traj_idxs.append([idx1, idx2])

    kappa = 1./np.load(msm_savedir + "/tica_ti.npy")[:M]

    X = np.zeros((M*P, R+1), float)
    d = np.zeros(M*P, float)

    print "calculating matrix elements..."
    Ntot = 0
    for n in range(len(traj_idxs)):
        starttime = time.time()

        print "traj: ", n+1
        sys.stdout.flush()
        idx1, idx2 = traj_idxs[n]

        # load tics
        psi_traj = []
        for i in range(M):
            # save TIC with indices of corresponding traj
            tic_saveas = msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, i+1)
            psi_traj.append(np.load(tic_saveas))
        psi_traj = np.array(psi_traj).T

        if len(psi_traj.shape) == 1:
            psi_traj = psi_traj.reshape((psi_traj.shape[0], 1))

        # calculate matrix elements
        start_idx = 0
        chunk_num = 1
        for chunk in md.iterload(trajnames[n], top=topfile, chunk=1000):
            print "    chunk: ", chunk_num
            sys.stdout.flush()
            chunk_num += 1

            # eigenfunction approximation
            Psi = psi_traj[start_idx:start_idx + chunk.n_frames,:]

            # calculate gradient of fixed and parametric potential terms
            grad_U0 = Ucg.gradient_U0(chunk, Psi)
            grad_U1 = Ucg.gradient_U1(chunk, Psi)

            # calculate test function values, gradient, and Laplacian
            grad_f, Lap_f = Ucg.test_funcs_gradient_and_laplacian(chunk, Psi) 
            test_f = Ucg.test_functions(chunk, Psi)

            # very useful einstein summation function to calculate
            # dot products with eigenvectors
            tempX = np.einsum("tm,tdr,tdp->mpr", Psi, -grad_U1, grad_f).reshape((M*P, R))
            #tempX2 = np.einsum("tm,tp->mp", Psi, test_f).reshape(M*P)

            tempZ = np.einsum("tm,td,tdp->mp", Psi, grad_U0, grad_f).reshape(M*P)
            tempY = (-1./beta)*np.einsum("tm,tp->mp", Psi, Lap_f).reshape(M*P)

            tempX2 = np.einsum("m,tm,tp->mp", kappa, Psi, test_f).reshape(M*P)

            #np.reshape(tempX, (M*P, R))
            X[:,:-1] += tempX
            X[:,-1] += tempX2
            d += tempZ + tempY

            start_idx += chunk.n_frames
            Ntot += chunk.n_frames

        min_calc_traj = (time.time() - starttime)
        print "calculation took: {:.4f} sec".format(min_calc_traj)
        sys.stdout.flush()

        #if n >= 5:
        #    break

    #X /= float(Ntot)
    #d /= float(Ntot)
    
    #"Ucg_eigenpair"
    #eg_savedir = msm_savedir + "/Ucg_eigenpair"
    eg_savedir = "run_" + str(run_idx) + "/" + cg_savedir 
    if not os.path.exists(eg_savedir):
        os.mkdir(eg_savedir)
    os.chdir(eg_savedir)

    np.save("X.npy", X)
    np.save("d.npy", d)

    with open("X_cond.dat", "w") as fout:
        fout.write(str(np.linalg.cond(X)))

    with open("Ntot.dat", "w") as fout:
        fout.write(str(Ntot))

    lstsq_soln = np.linalg.lstsq(X, d)
    np.save("coeff.npy", lstsq_soln[0])

    raise SystemExit

    traj = md.load(traj_name, top=topname)

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_validate

    import sklearn.model_selection

    cv_mean = []
    cv_std = []
    coeffs = []
    alphas = np.logspace(-10, 3, num=100)
    for i in range(len(alphas)):
        print i
        rdg = Ridge(alpha=alphas[i], fit_intercept=False)
        rdg = Ridge(alpha=1, fit_intercept=False)
        #X_train, X_test, y_train, y_test = train_test_split(G, frav, test_size=0.5, random_state=0)
        #model = rdg.fit(X_train, y_train)
        #cv_score.append(model.score(X_test, y_test))
        #coeffs.append(model.coef_)

        #model = rdg.fit(G, frav)
        #coeffs.append(model.coef_)

        #scores = cross_val_score(rdg, G, frav, cv=5)
        cv = cross_validate(rdg, G, frav, cv=5, return_estimator=True)

        cv_mean.append(scores.mean())
        cv_std.append(scores.std())
        

    train_sc, test_sc = sklearn.model_selection.validation_curve(rdg, G, frav, "alpha", np.logspace(-10, 3), cv=5)

    plt.figure()
    plt.errorbar(alphas, cv_mean, yerr=cv_std)

    plt.savefig("cv_score_vs_alpha.pdf")
    plt.savefig("cv_score_vs_alpha.png")

    plt.figure()
    plt.plot(alphas, cv_score)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"CV score")
    plt.savefig("cv_score_vs_alpha.pdf")
    plt.savefig("cv_score_vs_alpha.png")

