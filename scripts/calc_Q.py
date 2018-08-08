
import os
import glob
import numpy as np
#import matplotlib.pyplot as plt

import mdtraj as md

if __name__ == "__main__":
    pdb = md.load("c25_min_1.pdb")
    ply_idxs = pdb.topology.select("resname PLY")

    pairs = []
    for i in range(len(ply_idxs) - 1):
        for j in range(i + 3, len(ply_idxs)):
            pairs.append([ply_idxs[i], ply_idxs[j]])
    pairs = np.array(pairs)

    r0 = 0.4
    w = 0.1
    contact = lambda r: 0.5*(np.tanh(-(r - r0)/w) + 1)

    Q = []
    for chunk in md.iterload("c25_traj_1.dcd", top=pdb, atom_indices=ply_idxs):
        q = np.sum(contact(md.compute_distances(chunk, pairs)), axis=1)
        Q.append(q)
    Q = np.concatenate(Q)
    np.save("Q.npy", Q)
