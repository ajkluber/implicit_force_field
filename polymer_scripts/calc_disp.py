import os
import numpy as np

import mdtraj as md

def diag_idxs(n, k):
    rows, cols = np.diag_indices_from(np.empty((n, n)))
    diag = rows[:-k], cols[k:]
    return diag

if __name__ == "__main__":
    pdb = md.load("c25_min_1.pdb")
    ply_idxs = pdb.topology.select("resname PLY")

    subdir = "rij_dists" 
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    
    sep = range(1, len(ply_idxs))

    pairs = []
    for i in range(len(ply_idxs)):
        i_pairs = []
        for j in range(len(ply_idxs)):
            i_pairs.append([ply_idxs[i], ply_idxs[j]])
        pairs.append(i_pairs)
    pairs = np.array(pairs)

    import time
    #sep = [10]
    for k in sep:
        starttime = time.time()
        rmax = (k + 1)*0.12
        if rmax > 1.5:
            rmax = 1.5
        bin_edges = np.linspace(0.1, rmax, 100)
        mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

        my_hist = lambda data: np.histogram(data, bins=bin_edges)[0]

        k_idxs = diag_idxs(len(pairs), k)
        n_dists = len(pairs) - k
        all_hist = np.zeros((n_dists, len(mid_bin)), float)

        n_frames = 0
        # for each seq separation. calculate the distribution of distances.
        for chunk in md.iterload("c25_traj_1.dcd", top=pdb, atom_indices=ply_idxs):
            rij = md.compute_distances(chunk, pairs[k_idxs])

            all_hist += np.array(map(my_hist, rij.T))
            n_frames += chunk.n_frames

        np.save("{}/dist_{}.npy".format(subdir, k), all_hist)
        np.save("{}/pairs_{}.npy".format(subdir, k), pairs[k_idxs])
        np.savetxt("{}/n_frames_{}.dat".format(subdir, k), np.array([n_frames]))
        np.save("{}/bin_edges_{}.npy".format(subdir, k), bin_edges)

        stoptime = time.time()
        print("sep {}  took: {} min".format(k, (stoptime - starttime)/60.))

