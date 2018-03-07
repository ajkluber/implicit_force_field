import os
import numpy as np
import sympy
import matplotlib.pyplot as plt

import mdtraj as md
import simtk.unit as unit

import simulation.openmm as sop

if __name__ == "__main__":
    # diffusive motion:
    #  - Accelaration is much smaller than velocity. m*d^2x/dt << gamma*dx/dt
    #  - Velocity correlation function 

    n_beads = 25 

    topfile = "c25_min_1.pdb"
    trajfile = "c25_traj_1.dcd"
    traj = md.load(trajfile, top=topfile)

    gamma = 1.

    sigma_ply, eps_ply, mass_ply, bonded_params = sop.build_ff.toy_polymer_params()
    r0, kb, theta0, ka = bonded_params 

    s_frames = 1
    dt_frame = 0.2  # ps
    s = dt_frame*s_frames

    s_list = [1, 2, 5, 10, 40, 50, 100, 200, 500]
    n_col = 3
    fig, axes = plt.subplots(n_col, n_col, figsize=(n_col*4, n_col*4))
    for i in range(len(s_list)):
        skip = s_list[i]
        s = dt_frame*skip

        md2xdt2 = 37*(traj.xyz[2*skip:] - 2*traj.xyz[skip:-skip] + traj.xyz[:-2*skip])/(s**2)
        ydxdt = 37*gamma*(traj.xyz[skip:] - traj.xyz[:-skip])/s

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
    fig.savefig("finite_diff_vel_vs_acc_vs_s.pdf")
    fig.savefig("finite_diff_vel_vs_acc_vs_s.png")
    plt.show()

    #f_unit = 37.*unit.amu*unit.nanometer/(unit.picosecond**2)
    #e_unit = 37.*(unit.nanometer/unit.picosecond)**2

    #traj.xyz

