import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('subdir', type=str, help='Name.')
    args = parser.parse_args()

    name = args.name
    subdir = args.subdir

    savedir = "dih_dists" 

    os.chdir(subdir)
    if len(glob.glob("*/T_*")) > 0:
        # We are in directory above temps.
        Tpaths = glob.glob("*/T_*")
    else:
        Tpaths = glob.glob("T_*")

    cwd = os.getcwd()
    for i in range(len(Tpaths)):
        os.chdir(Tpaths[i])

        files = ["dih_dists.npy", "dih_idxs.npy", "bin_edges.npy"]
        all_files_exist = np.all([ os.path.exists(savedir + "/" + x) for x in files ])

        no_data = False
        if all_files_exist:
            os.chdir(savedir)
            all_n = np.load("dih_dists.npy")
            dih_idxs = np.load("dih_idxs.npy")
            bin_edges = np.load("bin_edges.npy")
            mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

            n_dih = len(dih_idxs)
            n_cols = int(np.round(np.sqrt(float(n_dih))))
            n_rows = n_cols
            if n_cols**2 < n_dih:
                n_rows = n_cols + 1

            fig, axes = plt.subplots(n_cols, n_rows, figsize=(4*n_cols, 4*n_rows), sharex=True, sharey=True)
            for i in range(n_dih):
                ax = axes[i / n_cols, i % n_cols]
                ax.fill_between(mid_bin, all_n[i,:])
                ax.grid(True, alpha=1, color='k', ls='--')
                ax.plot(mid_bin, all_n[i,:], color="b", lw=2)
                ax.set_title("({}, {}, {}, {})".format(dih_idxs[i,0], dih_idxs[i,1], dih_idxs[i,2], dih_idxs[i,3]), fontsize=10)
                #ax.set_yticks([])
                ax.set_yticklabels([])

            for i in range(n_cols):
                ax = axes[-1, i]
                ax.set_xlim(np.min(bin_edges), np.max(bin_edges))
                ax.set_xticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi])
                ax.set_xticklabels([r"$-\pi$",r"$\pi/2$","0",r"$\pi/2$",r"$\pi$"])

            big_ax = fig.add_subplot(111)
            big_ax.grid(False)
            #big_ax.set_axis_bgcolor('none')
            big_ax.set_facecolor('none')
            big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            big_ax.spines["top"].set_visible(False)
            big_ax.spines["bottom"].set_visible(False)
            big_ax.spines["right"].set_visible(False)
            big_ax.spines["left"].set_visible(False)
            big_ax.set_xlabel(r"Dihedral ($\phi$)", fontsize=24)
            big_ax.set_ylabel("Distribution", fontsize=24)
            #big_ax.set_title("Separation = " + str(k), fontsize=20)

            #fig.suptitle("Dihedrals", fontsize=26)
            fig.savefig("dists_dih.pdf")
            os.chdir("..")
        os.chdir(cwd)
