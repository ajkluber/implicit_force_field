import os
import glob
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

def orig_tic1_vs_time():
    # sort runs
    deltaT = 0.2
    run_idxs = [ (os.path.dirname(tname)).split("_")[-1] for tname in trajnames ]
    run_idxs = np.unique(run_idxs)

    fig, axes = plt.subplots(len(run_idxs), 1, figsize=(10, len(run_idxs)*4), sharex=True, sharey=True)
    maxx = 0
    for i in range(len(run_idxs)):
        n_run_trajs = np.sum([ "run_" + run_idxs[i] in tname for tname in trajnames ])

        temp_run_idxs = []
        for j in range(n_run_trajs):
            temp_run_idxs.append(trajnames.index("run_{}/c25_traj_{}.dcd".format(i + 1, j + 1)))
        run_cv = np.concatenate([ all_cv[x] for x in range(len(all_cv)) if x in temp_run_idxs ])

        if run_cv.shape[0] > maxx:
            maxx = run_cv.shape[0]

        ax = axes[i]
        x = deltaT*np.arange(0, len(run_cv))
        ax.annotate("Run " + run_idxs[i], xy=(0,0), xytext=(0.02, 0.85),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=18, bbox={"alpha":1, "edgecolor":"k", "facecolor":"w"})
        ax.plot(x, run_cv, lw=0.5)
        ax.set_xlim(0, deltaT*maxx)
        ax.set_ylim(-2, 2)
        ax.set_ylabel("Ref $\psi_2$")

    ax.set_xlabel("Time (ps)")
    fig.savefig("orig_tic1_vs_time.pdf")
    fig.savefig("orig_tic1_vs_time.png")

if __name__ == "__main__":
    #msm_savedir = "msm_dists"
    #os.chdir(msm_savedir)

    ticnames = glob.glob("run_*_TIC_1.npy")

    run_idxs = [ tname.split("_")[1] for tname in ticnames ]
    run_idxs = np.unique(run_idxs)
    run_idxs.sort()
    deltaT = 0.2

    fig, axes = plt.subplots(len(run_idxs), 1, figsize=(10, len(run_idxs)*4), sharex=True, sharey=True)
    maxx = 0
    for i in range(len(run_idxs)):
        n_run_trajs = len(glob.glob("run_{}_*_TIC_1.npy".format(run_idxs[i])))

        run_cv = []
        for j in range(n_run_trajs):
            run_cv.append(np.load("run_{}_{}_TIC_1.npy".format(run_idxs[i], j + 1)))
        run_cv = np.concatenate(run_cv)

        if run_cv.shape[0] > maxx:
            maxx = run_cv.shape[0]

        ax = axes[i]
        x = deltaT*np.arange(0, len(run_cv))
        ax.annotate("Run " + run_idxs[i], xy=(0,0), xytext=(0.02, 0.85),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=18, bbox={"alpha":1, "edgecolor":"k", "facecolor":"w"})
        ax.plot(x, run_cv, lw=0.5)
        ax.set_xlim(0, deltaT*maxx)
        ax.set_ylim(-2, 2)
        ax.set_ylabel("Ref $\psi_2$")

    ax.set_xlabel("Time (ps)")
    fig.savefig("psi2_vs_time.pdf")
    #fig.savefig("psi2_vs_time.png")
