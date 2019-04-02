from __future__ import print_function
import os
import glob
import sys
import shutil
import time
import argparse
import numpy as np
import scipy.interpolate
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

import simtk.unit as unit
import simtk.openmm as omm
import simtk.openmm.app as app

import mdtraj as md

import simulation.openmm as sop

import implicit_force_field.polymer_scripts.util as util

if __name__ == "__main__":
    # 1. Calculate original and new TICs for cg sims
    # 2. Calculate PMF(psi), <E>(psi), and S(psi)
    # 3. Plot
    
    savedir = "avgE_vs_orig_tic1"

    tname = "run_[1-9]*/" + name + "_traj_*.dcd"

    trajnames = glob.glob(name + "_traj_cent_*.dcd")


    traj = md.load("c25_traj_1.dcd", top="c25_noslv_min.pdb")
    bond_r = md.compute_distances(traj, np.array([[i , i + 1] for i in range(24) ]))
    angle_theta = md.compute_angles(traj, np.array([[i , i + 1, i + 2] for i in range(23) ]))

    pair_idxs = []
    for i in range(n_beads - 1):
        for j in range(i + 4, n_beads):
            pair_idxs.append([i, j])
    pair_idxs = np.array(pair_idxs)
    rij = md.compute_distances(traj, pair_idxs)

    Uex_md = np.sum(0.5*((0.373/rij)**12), axis=1)
    Ub_md = np.sum(0.5*334720.0*((bond_r - 0.153)**2), axis=1)
    Ua_md = np.sum(0.5*462.0*((angle_theta - 1.938)**2), axis=1)

    xyz_traj = np.reshape(traj.xyz, (-1, 75))
    cv_traj = Ucg.calculate_cv(xyz_traj)

    U0 = Ucg.potential_U0(xyz_traj, cv_traj, sumterms=False)
    #Ub, Ua, Uex = U0
    Ub, Ua = U0

    U_k = Ucg.potential_U1(xyz_traj, cv_traj)
    Uex = coeff[0]*Uex_md
    if using_cv:
        U1_calc = np.einsum("k,tk->t", coeff[1:], U_k[:,1:])
        U1_label = "CV"
    else:
        U1_calc = np.einsum("k,tk->t", coeff, U_k)
        U1_label = "Pair"
