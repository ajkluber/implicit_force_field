import os
import glob
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import mdtraj as md

import simulation.openmm as sop
import implicit_force_field as iff

if __name__ == "__main__":
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

    # scaling the terms of the potential reduces the condition number of the
    # force matrix by several orders of magnitude.

    # create polymer model with free parameters
    Ucg = iff.basis_library.PolymerModel(n_beads)
    #Ucg._assign_harmonic_bonds(r0_nm, scale_factor=kb_kj)
    #Ucg._assign_harmonic_angles(theta0_rad, scale_factor=ka_kj)
    #Ucg._assign_LJ6(sigma_ply_nm, scale_factor=eps_ply_kj)
    Ucg._assign_harmonic_bonds(r0_nm)
    Ucg._assign_harmonic_angles(theta0_rad)
    Ucg._assign_LJ6(sigma_ply_nm)

    # add test functions
    
    Ucg._assign_bond_funcs([r0_nm], [0.3])
    Ucg._assign_angle_funcs([theta0_rad], [4])
    #Ucg._assign_pairwise_funcs(sigma_ply_nm)


    ##########################################################
    # calculate integrated sindy (eigenpair) matrix equation.
    ########################################################## 
    msm_savedir = "msm_dih_dists"

    # load tics
    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]

    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd") 
    traj_idxs = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        traj_idxs.append([idx1, idx2])

    M = 3
    D = Ucg.n_dof 
    R = Ucg.n_params
    P = Ucg.n_test_funcs
    kappa = 1./np.load(msm_savedir + "/tica_ti.npy")[:M]

    X = np.zeros((M*P, R+1), float)
    d = np.zeros(M*P, float)

    Ntot = 0
    for n in range(len(traj_idxs)):
        starttime = time.time()

        print "traj: ", n+1
        idx1, idx2 = traj_idxs[n]

        # load tics
        psi_traj = []
        for i in range(M):
            # save TIC with indices of corresponding traj
            tic_saveas = msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, i+1)
            psi_traj.append(np.load(tic_saveas))
        psi_traj = np.array(psi_traj).T

        # calculate matrix elements
        start_idx = 0
        chunk_num = 1
        for chunk in md.iterload(trajnames[n], top=topfile, chunk=10000):
            print "    chunk: ", chunk_num
            chunk_num += 1

            # calculate model quantities. Potential and test functions gradients, etc.
            test_f = Ucg.calculate_test_funcs(chunk)
            grad_U = Ucg.calculate_U_gradient(chunk)
            grad_U0 = Ucg.calculate_U0_gradient(chunk)
            grad_f = Ucg.calculate_test_func_gradient(chunk) 
            Lap_f = Ucg.calculate_test_func_laplacian(chunk) 

            #psi_i = psi_trajs[0,start_idx:start_idx + chunk.n_frames]
            Psi = psi_traj[start_idx:start_idx + chunk.n_frames,:]

            # dot products with eigenvectors
            tempX = np.einsum("tm,tdr,tdp->mpr", Psi, -grad_U, grad_f).reshape((M*P, R))
            tempZ = np.einsum("tm,td,tdp->mp", Psi, grad_U0, grad_f).reshape(M*P)
            tempY = (-1./beta)*np.einsum("tm,tp->mp", Psi, Lap_f).reshape(M*P)
            tempX2 = np.einsum("tm,tp->mp", Psi, test_f).reshape(M*P)

            tempX2 = np.einsum("m,tm,tp->mp", kappa, Psi, test_f).reshape(M*P)

            #np.reshape(tempX, (M*P, R))
            X[:,:-1] += tempX
            X[:,-1] += tempX2
            d += tempZ + tempY

            start_idx += chunk.n_frames
            Ntot += chunk.n_frames

        min_calc_traj = (time.time() - starttime)
        print "calculation took: {:.4f} sec".format(min_calc_traj)
    
    #c_soln = np.linalg.lstsq(X, d)[0]

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

