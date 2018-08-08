import os
import numpy as np

import mdtraj as md

if __name__ == "__main__":
    pdb = md.load("c25_min_1.pdb")
    ply_idxs = pdb.topology.select("resname PLY")

    subdir = "dih_dists" 
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    
    dih_idxs = []
    for i in range(len(ply_idxs) - 3):
        idx = ply_idxs[i]
        dih_idxs.append([idx, idx + 1, idx + 2, idx + 3])
    dih_idxs = np.array(dih_idxs)

    bin_edges = np.linspace(-np.pi, np.pi, 100)
    mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

    my_hist = lambda data: np.histogram(data, bins=bin_edges)[0]

    n_ang = len(dih_idxs)
    all_hist = np.zeros((n_ang, len(mid_bin)), float)

    n_frames = 0
    # for each dihedral. calculate the distribution
    for chunk in md.iterload("c25_traj_1.dcd", top=pdb, atom_indices=ply_idxs):
        phi = md.compute_dihedrals(chunk, dih_idxs)
        all_hist += np.array(map(my_hist, phi.T))
        n_frames += chunk.n_frames

    np.save("{}/dih_dists.npy".format(subdir), all_hist)
    np.save("{}/dih_idxs.npy".format(subdir), dih_idxs)
    np.savetxt("{}/n_frames.dat".format(subdir), np.array([n_frames]))
    np.save("{}/bin_edges.npy".format(subdir), bin_edges)


def calc_for_dih22():
    pdb = md.load("c25_min_1.pdb")
    ply_idxs = pdb.topology.select("resname PLY")

    subdir = "dih_dists" 
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    
    dih_idxs = np.array([21,22,23,24])

    bin_edges = np.linspace(-np.pi, np.pi, 100)
    mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

    all_hist = np.zeros(len(mid_bin), float)

    n_frames = 0
    # for each dihedral. calculate the distribution
    for chunk in md.iterload("c25_traj_1.dcd", top=pdb, atom_indices=ply_idxs):
        phi = md.compute_dihedrals(chunk, np.array([[0,1,2,3]]))[:,0]
        all_hist += np.histogram(phi, bins=bin_edges)[0]

    np.save("{}/dih_22.npy".format(subdir), all_hist)
