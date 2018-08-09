import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#import mdtraj as md

if __name__ == "__main__":
    #pdb = md.load("c25_min_1.pdb")
    #ply_idxs = pdb.topology.select("resname PLY")

    subdir = "dih_dists" 
    if not os.path.exists("plots"):
        os.mkdir("plots")

    all_n = np.load("{}/dih_dists.npy".format(subdir))
    dih_idxs = np.load("{}/dih_idxs.npy".format(subdir))
    n_frames = np.loadtxt("{}/n_frames.dat".format(subdir))
    bin_edges = np.load("{}/bin_edges.npy".format(subdir))
    mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])
    n_col = int(np.ceil(np.sqrt(float(all_n.shape[0]))))

    for k in range(len(all_n)):
        fig, axes = plt.subplots(n_col, n_col, figsize=(4*n_col, 4*n_col), sharex=True, sharey=True)
        for i in range(all_n.shape[0]):
            ax = axes[i / n_col, i % n_col]
            ax.fill_between(mid_bin, all_n[i,:])
            ax.plot(mid_bin, all_n[i,:], color="b", lw=2)
            ax.set_title("({}, {}, {}, {})".format(dih_idxs[i,0], dih_idxs[i,1], dih_idxs[i,2], dih_idxs[i,3]), fontsize=10)
            ax.set_yticks([])

        for i in range(n_col):
            ax = axes[-1, i]
            ax.set_xlim(np.min(bin_edges), np.max(bin_edges))
            ax.set_xticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi])
            ax.set_xticklabels([r"$-\pi$",r"$\pi/2$","0",r"$\pi/2$",r"$\pi$"])
        #    if k < 3:
        #        xinc = 0.5
        #    elif 3 <= k < 8:
        #        xinc = 1
        #    else:
        #        xinc = 2
        #    ax.set_xticks(np.arange(1, 10*np.max(bin_edges) + 0.1, xinc))

        big_ax = fig.add_subplot(111)
        big_ax.grid(False)
        big_ax.set_axis_bgcolor('none')
        big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        big_ax.spines["top"].set_visible(False)
        big_ax.spines["bottom"].set_visible(False)
        big_ax.spines["right"].set_visible(False)
        big_ax.spines["left"].set_visible(False)
        big_ax.set_xlabel(r"Dihedral ($\phi$)", fontsize=24)
        big_ax.set_ylabel("Distribution", fontsize=24)
        #big_ax.set_title("Separation = " + str(k), fontsize=20)

        #fig.suptitle("Dihedrals", fontsize=26)
        fig.savefig("plots/dists_dih.pdf")
