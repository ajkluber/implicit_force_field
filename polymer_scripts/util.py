from __future__ import print_function, absolute_import
import numpy as np

import simtk.unit as unit
import simtk.openmm.app as app

import simulation.openmm as sop
import implicit_force_field as iff

def Ucg_dirname(method, M, using_U0, fix_back, fix_exvol, bond_cutoff,
        using_cv, n_cv_basis_funcs=None, n_cv_test_funcs=None,
        n_pair_gauss=None, a_coeff=None):

    if method == "force-matching":
        cg_savedir = "Ucg_FM"
    elif method == "eigenpair":
        cg_savedir = "Ucg_EG"
    else:
        raise ValueError("Method must be force-matching or eigenpair")

    if using_U0:
        if fix_back:
            cg_savedir += "_fixback"

        if fix_exvol:
            cg_savedir += "_fixexvol"
    else:
        cg_savedir += "_noU0"

    if using_cv:
        cg_savedir += "_CV_{}_{}_{}".format(M, n_cv_basis_funcs, n_cv_test_funcs)
    else:
        cg_savedir += "_pair_{}".format(n_pair_gauss)

    if using_U0 or not using_cv:
        cg_savedir += "_bondcut_{}".format(bond_cutoff)

    if method == "eigenpair" and (not a_coeff is None):
        #cg_savedir += "_fixed_a"
        cg_savedir += "_fixed_a_{:.2e}".format(a_coeff)

    return cg_savedir 

def create_polymer_Ucg(msm_savedir, n_beads, M, beta, fix_back, fix_exvol,
        using_cv, using_D2, n_cv_basis_funcs, n_cv_test_funcs, n_pair_gauss,
        bond_cutoff, a_coeff=None, n_cvs=None):

    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    r0, kb, theta0, ka = bonded_params

    if "PL" not in app.element.Element._elements_by_symbol:
        app.element.polymer = app.element.Element(200, "Polymer", "Pl", mass_ply)

    sigma_ply_nm = sigma_ply/unit.nanometer
    #r0_wca_nm = sigma_ply_nm*(2**(1./6))
    eps_ply_kj = eps_ply/unit.kilojoule_per_mole
    kb_kj = kb/(unit.kilojoule_per_mole/(unit.nanometer**2))
    ka_kj = (ka/(unit.kilojoule_per_mole/(unit.radian**2)))
    theta0_rad = theta0/unit.radian
    r0_nm = r0/unit.nanometer


    print("creating Ucg with...")
    # coarse-grain polymer potential with free parameters
    Ucg = iff.basis_library.PolymerModel(n_beads, beta, using_cv=using_cv, using_D2=using_D2, a_coeff=a_coeff)

    if fix_back:
        print("  fixed bond and angle potentials")
        Ucg.harmonic_bond_potentials(r0_nm, scale_factor=kb_kj, fixed=True)
        Ucg.harmonic_angle_potentials(theta0_rad, scale_factor=ka_kj, fixed=True)
        #Ucg.LJ6_potentials(sigma_ply_nm, scale_factor=eps_ply_kj)

    if fix_exvol:
        print("  fixed excluded volume potentials")
        Ucg.inverse_r12_potentials(sigma_ply_nm, scale_factor=0.5, fixed=True,
                bond_cutoff=bond_cutoff)
    else:
        print("  parametric excluded volume potentials")
        Ucg.inverse_r12_potentials(sigma_ply_nm, scale_factor=0.5, fixed=False,
                bond_cutoff=bond_cutoff)

    if using_cv:
        print("  collective variable potentials")
        # centers of test functions in collective variable (CV) space
        temp_cv_r0 = np.load(msm_savedir + "/psi1_mid_bin.npy")[1:-1]
        cv_r0 = np.linspace(temp_cv_r0.min(), temp_cv_r0.max(), n_cv_basis_funcs)
        cv_w_basis = 2*np.abs(cv_r0[1] - cv_r0[0])*np.ones(len(cv_r0), float)
        cv_r0_basis = cv_r0.reshape((-1, 1))

        cv_r0 = np.linspace(temp_cv_r0.min(), temp_cv_r0.max(), n_cv_test_funcs)
        cv_w_test = 2*np.abs(cv_r0[1] - cv_r0[0])*np.ones(len(cv_r0), float)
        cv_r0_test = cv_r0.reshape((-1, 1))
             
        ply_idxs = np.arange(n_beads)
        pair_idxs = []
        for i in range(len(ply_idxs) - 1):
            for j in range(i + 4, len(ply_idxs)):
                pair_idxs.append([ply_idxs[i], ply_idxs[j]])
        pair_idxs = np.array(pair_idxs)

        if n_cvs is None:
            n_cvs = M

        cv_coeff = np.load(msm_savedir + "/tica_eigenvects.npy")[:,:n_cvs]
        cv_mean = np.load(msm_savedir + "/tica_mean.npy")

        Ucg.linear_collective_variables(["dist"], pair_idxs, cv_coeff, cv_mean)
        Ucg.gaussian_cv_test_funcs(cv_r0_test, cv_w_test)
        Ucg.gaussian_cv_potentials(cv_r0_basis, cv_w_basis)

    else:
        print("  pairwise gaussian potentials")
        cv_r0_basis = None
        gauss_r0_nm = np.linspace(0.3, 1, n_pair_gauss)
        gauss_sigma = gauss_r0_nm[1] - gauss_r0_nm[0]
        gauss_w_nm = gauss_sigma*np.ones(len(gauss_r0_nm))
        Ucg.gaussian_pair_potentials(gauss_r0_nm, gauss_w_nm, scale_factor=10)
        Ucg.gaussian_bond_test_funcs([r0_nm], [0.3])
        Ucg.vonMises_angle_test_funcs([theta0_rad], [4])
        Ucg.gaussian_pair_test_funcs(gauss_r0_nm, gauss_w_nm)

    return Ucg, cv_r0_basis, cv_r0_test

def create_Ucg_collective_variable(msm_savedir, n_beads, n_cvs, beta,
        using_cv, using_D2, bond_cutoff, a_coeff=None):

    if "PL" not in app.element.Element._elements_by_symbol:
        sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
        app.element.polymer = app.element.Element(200, "Polymer", "Pl", mass_ply)

    print("creating Ucg with...")
    # coarse-grain polymer potential with free parameters
    Ucg = iff.basis_library.PolymerModel(n_beads, beta, using_cv=using_cv, using_D2=using_D2, a_coeff=a_coeff)

    print("  collective variable potentials")
    # centers of test functions in collective variable (CV) space
    ply_idxs = np.arange(n_beads)
    pair_idxs = []
    for i in range(len(ply_idxs) - 1):
        for j in range(i + 4, len(ply_idxs)):
            pair_idxs.append([ply_idxs[i], ply_idxs[j]])
    pair_idxs = np.array(pair_idxs)

    cv_coeff = np.load(msm_savedir + "/tica_eigenvects.npy")[:,:n_cvs]
    cv_mean = np.load(msm_savedir + "/tica_mean.npy")

    Ucg.linear_collective_variables(["dist"], pair_idxs, cv_coeff, cv_mean)

    return Ucg
