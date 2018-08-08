import os
import glob
import numpy as np
#import matplotlib.pyplot as plt

import mdtraj as md

if __name__ == "__main__":
    pdb = md.load("c25_min_1.pdb")
    ply_idxs = pdb.topology.select("resname PLY")

    Rg = []
    for chunk in md.iterload("c25_traj_1.dcd", top=pdb, atom_indices=ply_idxs):
        rg = md.compute_rg(chunk)
        Rg.append(rg)
    Rg = np.concatenate(Rg)
    np.save("Rg.npy", Rg)

    #plt.figure()
    #plt.plot(time, Rg)
    #plt.xlabel("Time (ps)")
    #plt.ylabel("Radius gyration (nm)")
    #plt.title(name + " " + "T={:.2f}".format(T) + " " + rundir)
    #plt.ylim(0, 1)
    #plt.savefig("Rg_vs_t.pdf")
