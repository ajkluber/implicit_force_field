import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#import mdtraj as md

if __name__ == "__main__":
    #pdb = md.load("c25_min_1.pdb")
    #ply_idxs = pdb.topology.select("resname PLY")

    subdir = "rij_dists" 

    if not os.path.exists("plots"):
        os.mkdir("plots")

    # for separation plot all distributions
    #sep = [10]
    #sep = [24]
    sep = range(1, 25)
    for k in sep:
        print("sep: " + str(k))
        all_n = np.load("{}/dist_{}.npy".format(subdir, k))
        pairs_k = np.load("{}/pairs_{}.npy".format(subdir, k))
        n_frames_k = np.loadtxt("{}/n_frames_{}.dat".format(subdir, k))
        bin_edges = np.load("{}/bin_edges_{}.npy".format(subdir, k))

        mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

        n_col = int(np.ceil(np.sqrt(float(all_n.shape[0]))))

        if n_col > 1:
            fig, axes = plt.subplots(n_col, n_col, figsize=(4*n_col, 4*n_col), sharex=True, sharey=True)
            for i in range(all_n.shape[0]):
                ax = axes[i / n_col, i % n_col]
                ax.fill_between(10*mid_bin, all_n[i,:])
                ax.plot(10*mid_bin, all_n[i,:], color="b", lw=2)
                ax.set_title("({}, {})".format(pairs_k[i,0], pairs_k[i,1]))
                ax.set_yticks([])

            for i in range(n_col):
                ax = axes[-1, i]
                ax.set_xlim(10*np.min(bin_edges), 10*np.max(bin_edges))
                if k < 3:
                    xinc = 0.5
                elif 3 <= k < 8:
                    xinc = 1
                else:
                    xinc = 2
                ax.set_xticks(np.arange(1, 10*np.max(bin_edges) + 0.1, xinc))

            big_ax = fig.add_subplot(111)
            big_ax.grid(False)
            big_ax.set_axis_bgcolor('none')
            big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            big_ax.spines["top"].set_visible(False)
            big_ax.spines["bottom"].set_visible(False)
            big_ax.spines["right"].set_visible(False)
            big_ax.spines["left"].set_visible(False)
            big_ax.set_xlabel(r"Distance ($\AA$)", fontsize=24)
            big_ax.set_ylabel("Distribution", fontsize=24)
            #big_ax.set_title("Separation = " + str(k), fontsize=20)
        else:
            fig = plt.figure()
            plt.plot(10*mid_bin, all_n[0,:], color="b", lw=2)
            plt.fill_between(10*mid_bin, all_n[0,:])
            plt.xlabel("Distance (nm)")
            plt.ylabel("Distribution")

        fig.suptitle("Separation = " + str(k), fontsize=26)
        fig.savefig("plots/dists_{}.pdf".format(k))
