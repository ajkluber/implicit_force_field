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
    n_beads = 25
    name = "c" + str(n_beads)
    T = 300
    kb = 0.0083145
    beta = 1./(kb*T)

    #msm_savedir = "msm_dih_dists"
    msm_savedir = "msm_dists"

    M = 1   # number of eigenvectors to use

    cg_savedir = "Ucg_eigenpair_1D"

    psi_hist = np.load(msm_savedir + "/psi1_n.npy")
    cv_r0 = np.load(msm_savedir + "/psi1_mid_bin.npy")
    cv_w = np.abs(cv_r0[1] - cv_r0[0])*np.ones(len(cv_r0), float)
    #cv_r0 = np.array([ [cv_r0[i]] for i in range(len(cv_r0)) ])
    #cv_r0 = cv_r0.reshape((len(cv_r0),1))

    print "creating Ucg..."
    # coarse-grain polymer potential with free parameters
    Ucg = iff.basis_library.OneDimensionalModel(1)
    Ucg.add_Gaussian_drift_basis(cv_r0, cv_w)
    Ucg.add_Gaussian_noise_basis(cv_r0, cv_w)
    Ucg.add_Gaussian_test_functions(cv_r0, cv_w)

    n_a = len(Ucg.a_funcs[1])
    n_b = len(Ucg.b_funcs[1])
    R = n_a + n_b           # number of free model parameters
    P = len(Ucg.f_funcs)    # number of test functions

    ##########################################################
    # calculate integrated sindy (eigenpair) matrix equation.
    ########################################################## 
    #topfile = glob.glob("run_{}/".format(run_idx) + name + "_min_cent.pdb")[0]
    #trajnames = glob.glob("run_{}/".format(run_idx) + name + "_traj_cent_*.dcd") 
    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]
    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd") 
    traj_idxs = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        traj_idxs.append([idx1, idx2])

    kappa = 1./np.load(msm_savedir + "/tica_ti.npy")[0]

    X = np.zeros((P, R), float)
    d = np.zeros(P, float)

    print "calculating matrix elements..."
    Ntot = 0
    for n in range(len(traj_idxs)):
        print "traj: ", n+1
        sys.stdout.flush()
        idx1, idx2 = traj_idxs[n]
        Psi = np.load(msm_savedir + "/run_{}_{}_TIC_1.npy".format(idx1, idx2))

        # matrix elements 
        b1 = Ucg.evaluate_parametric_drift(Psi)
        a1 = Ucg.evaluate_parametric_noise(Psi)

        test_f = Ucg.test_functions(Psi)
        grad_f = Ucg.gradient_test_functions(Psi) 
        Lap_f = Ucg.laplacian_test_functions(Psi) 

        temp_X1 = np.einsum("t,tr,tp->pr", Psi, b1, grad_f)
        temp_X2 = (-1/beta)*np.einsum("t,tr,tp->pr", Psi, a1, Lap_f)

        temp_d = kappa*np.einsum("t,tp->p", Psi, test_f)

        X[:,:n_b] += temp_X1
        X[:,n_b:] += temp_X2
        d += temp_d

        Ntot += Psi.shape[0]

    X /= float(Ntot)
    d /= float(Ntot)
    
    #"Ucg_eigenpair"
    if not os.path.exists(cg_savedir):
        os.mkdir(cg_savedir)
    os.chdir(cg_savedir)

    np.save("X.npy", X)
    np.save("d.npy", d)

    with open("X_cond.dat", "w") as fout:
        fout.write(str(np.linalg.cond(X)))

    with open("Ntot.dat", "w") as fout:
        fout.write(str(Ntot))

    lstsq_soln = np.linalg.lstsq(X, d)
    np.save("coeff.npy", lstsq_soln[0])

    raise SystemExit


    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['mathtext.rm'] = 'serif'
    import matplotlib.pyplot as plt
    import sympy 
    x_sym = sympy.symbols("x")

    pmf = -np.log(psi_hist)
    pmf -= pmf.min()

    coeff = lstsq_soln[0]
    b_coeff = coeff[:n_b]
    a_coeff = coeff[n_b:]

    #r = np.linspace(min(cv_r0[2:]), max(cv_r0[:-2]), 200)
    r = np.linspace(-1.1, 1.1, 200)

    drift = np.zeros(len(r), float)
    for i in range(len(b_coeff)):
        drift += b_coeff[i]*Ucg.b_scale_factors[1][i]*Ucg.b_funcs[1][i](r)

    noise = np.zeros(len(r), float)
    for i in range(len(a_coeff)):
        noise += a_coeff[i]*Ucg.a_scale_factors[1][i]*Ucg.a_funcs[1][i](r)


    d_noise = np.zeros(len(r), float)
    for i in range(len(a_coeff)):
        da_temp = sympy.lambdify(x_sym, Ucg.a_sym[1][i].diff(x_sym), modules="numpy")(r)
        d_noise += a_coeff[i]*Ucg.a_scale_factors[1][i]*da_temp

    dF = (1/noise)*((1/beta)*d_noise - drift)

    plt.figure()
    plt.plot(r, -dF)
    plt.xlabel(r"TIC1 $\psi_1$")
    plt.ylabel(r"Mean force $-\nabla W(\psi_1)$")
    plt.ylim(-100, 100)
    plt.savefig("grad_F_100.pdf")
    plt.savefig("grad_F_100.png")

    xmin, xmax = min(cv_r0), max(cv_r0)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 15))
    ax1.plot(cv_r0, pmf)
    ax1.set_ylabel(r"Traj PMF $-\log(P(\psi_1))$")
    ax1.set_ylim(0, 4)

    ax2.plot(r, drift)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylabel(r"Drift $b(\psi_1)$")

    ax3.plot(r, noise)
    ax3.set_xlim(xmin, xmax)
    ax3.set_xlabel(r"TIC1 $\psi_1$")
    ax3.set_ylabel(r"Noise $a(\psi_1)$")

    fig.savefig("drift_noise_1D_100.pdf")
    fig.savefig("drift_noise_1D_100.png")
