import os
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
    #Ucg.harmonic_bond_potentials(r0_nm, scale_factor=kb_kj)
    #Ucg.harmonic_angle_potentials(theta0_rad, scale_factor=ka_kj)
    #Ucg.LJ6_potentials(sigma_ply_nm, scale_factor=eps_ply_kj)
    Ucg.harmonic_bond_potentials(r0_nm)
    Ucg.harmonic_angle_potentials(theta0_rad)
    Ucg.LJ6_potentials(sigma_ply_nm)

    raise SystemExit

    fsim = np.loadtxt(name + "_forces_{}.dat".format(traj_idx))
    frav = fsim.ravel()


    traj = md.load(traj_name, top=topname)
    G = Ucg.calculate_parametric_forces(traj)

    c_lstsq = np.linalg.lstsq(G, frav)

    raise SystemExit

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

