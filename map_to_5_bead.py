import os
import time
import numpy as np
import sympy
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold

import mdtraj as md
import simtk.unit as unit

import simulation.openmm as sop

if __name__ == "__main__":
    n_beads = 25 
    n_dim = 3*n_beads

    #topfile = "c25_min_ply_1.pdb"
    #trajfile = "c25_traj_ply_1.dcd"
    topfile = "c25_min_1.pdb"
    trajfile = "c25_traj_1.dcd"


    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    r0, kb, theta0, ka = bonded_params 

    # remove units from paramters
    sigma_ply_nm = sigma_ply/unit.nanometer
    r0_wca_nm = sigma_ply_nm*(2**(1./6))
    eps_ply_kj = eps_ply/unit.kilojoule_per_mole
    kb_kj = kb/(unit.kilojoule_per_mole/(unit.nanometer**2))
    ka_kj = (ka/(unit.kilojoule_per_mole/(unit.radian**2)))
    theta0_rad = theta0/unit.radian
    r0_nm = r0/unit.nanometer


    # mapping operator. coarsen 5 beads to one
    M_R = np.zeros((5, 25), float)
    for i in range(len(M_R)):
        M_R[i, np.arange(5) + i*5] = 1./5

    mapping = lambda xyz: np.dot(M_R, xyz)

    traj = md.load(trajfile, top=topfile)
    new_xyz = np.array(map(mapping, traj.xyz))

    #for chunk in md.iterload(trajfile, top=topfile, chunk=1000):
    #    new_xyz = np.array(map(mapping, chunk.xyz))
    #    break


    dt_frame = 0.2  # ps
    s_list = [1, 2, 5, 10, 40, 50, 100, 200, 500]
    n_col = 3
    fig, axes = plt.subplots(n_col, n_col, figsize=(n_col*4, n_col*4))
    for i in range(len(s_list)):
        skip = s_list[i]
        s = dt_frame*skip

        md2xdt2 = (new_xyz[2*skip:] - 2*new_xyz[skip:-skip] + new_xyz[:-2*skip])/(s**2)
        ydxdt = (new_xyz[skip:] - new_xyz[:-skip])/s

        ax = axes[i / n_col, i % n_col]
        if i == 0:
            _ = ax.hist(md2xdt2.ravel(), bins=200, histtype="stepfilled", alpha=0.5, color="k", label="Acceleration")
            _ = ax.hist(ydxdt.ravel(), bins=200, histtype="stepfilled", alpha=0.5, color="b", label="Velocity")
        else:
            _ = ax.hist(md2xdt2.ravel(), bins=200, histtype="stepfilled", color="k", alpha=0.5)
            _ = ax.hist(ydxdt.ravel(), bins=200, histtype="stepfilled", color="b", alpha=0.5)
        ax.set_title("s = {} ps".format(s), fontsize=18)
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig("coarse5_finite_diff_vel_vs_acc_vs_s.pdf")
    fig.savefig("coarse5_finite_diff_vel_vs_acc_vs_s.png")
    plt.show()




