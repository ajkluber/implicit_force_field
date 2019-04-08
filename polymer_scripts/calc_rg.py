import os
import time
import glob
import argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

import mdtraj as md

def get_rg_for_run(name, ply_idxs, pdb, use_cent, recalc):

    topfile, trajnames = get_trajnames(name, use_cent)
    rg_for_run = []
    for j in range(len(trajnames)):
        idx = j + 1
        if use_cent:
            tname = name + "_traj_cent_" + str(idx) + ".dcd"
        else:
            tname = name + "_traj_" + str(idx) + ".dcd"

        rg_name = "rg_{}.npy".format(idx)
        if not os.path.exists(rg_name) or recalc:
            if not os.path.exists(tname):
                raise IOError(tname + " does not exist!")

            last_change = np.abs(os.path.getmtime(tname) - time.time())/60.
            if last_change > 5:
                # only calculate if traj has been modified in last five minutes.
                # this is meant to check if traj is still running.
                Rg = []
                for chunk in md.iterload(tname, top=pdb, atom_indices=ply_idxs):
                    rg = md.compute_rg(chunk)
                    Rg.append(rg)
                Rg = np.concatenate(Rg)
                print("  " + rg_name)
                np.save(rg_name, Rg)
            else:
                Rg = None
        else:
            Rg = np.load(rg_name)
        if not (Rg is None):
            rg_for_run.append(Rg)
    return rg_for_run

def get_rg_for_all_runs(name, use_cent, recalc):
    """Returns list of Rg for each run directory, concatenates subtrajectories"""

    rundirs = glob.glob("run_[1-9]*")

    all_rg_files_exist = rg_files_exist(name, use_cent=use_cent)

    if all_rg_files_exist and not recalc:
        print("All rg files exist, don't have to calculate")

    if use_cent:
        topfile = glob.glob(os.getcwd() + "/run_[1-9]*/" + name + "_min_cent.pdb")[0]
    else:
        topfile = glob.glob(os.getcwd() + "/run_[1-9]*/" + name + "_noslv_min.pdb")[0]

    pdb = md.load(topfile)
    ply_idxs = pdb.top.select("name PL") 

    all_rg = [] 
    for i in range(len(rundirs)):
        os.chdir(rundirs[i])
        rg_for_run = get_rg_for_run(name, ply_idxs, pdb, use_cent, recalc)
    
        if len(rg_for_run) > 0:
            all_rg.append(np.concatenate(rg_for_run))
        else:
            print("No rg for " + rundirs[i] + " might be running still")
        os.chdir("..")

    return all_rg

def rg_files_exist(name, use_cent=False):

    if use_cent:
        trajnames = glob.glob("run_[1-9]*/{}_traj_cent_[1-9]*.dcd".format(name))
    else:
        trajnames = glob.glob("run_[1-9]*/{}_traj_[1-9]*.dcd".format(name))

    rg_files_exist = []
    for i in range(len(trajnames)):
        rundir = os.path.dirname(trajnames[i]) 
        idx = (trajnames[i].split("_")[-1])[:-4]
        rg_files_exist.append(os.path.exists(rundir + "/rg_{}.npy".format(idx)))
    return np.all(rg_files_exist)

def plot_vs_time(all_rg):

    deltaT = 0.2
    fig, axes = plt.subplots(len(all_rg), 1, figsize=(10, len(all_rg)*4), sharex=True, sharey=True)
    maxx = 0
    for i in range(len(all_rg)):
        run_rg = all_rg[i]

        if run_rg.shape[0] > maxx:
            maxx = run_rg.shape[0]

        ax = axes[i]
        x = deltaT*np.arange(0, len(run_rg))
        ax.annotate("Run " + str(i + 1), xy=(0,0), xytext=(0.02, 0.85),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=18, bbox={"alpha":1, "edgecolor":"k", "facecolor":"w"})
        ax.plot(x, run_rg, lw=0.5)
        ax.set_xlim(0, deltaT*maxx)
        ax.set_ylim(0.3, 0.8)
        ax.set_ylabel("$R_g$ (nm)")

    ax.set_xlabel("Time (ps)")
    fig.savefig("rg_vs_time.pdf")
    #fig.savefig("rg_vs_time.png")

def get_trajnames(name, use_cent):
    if use_cent:
        trajnames = glob.glob(name + "_traj_cent_[1-9]*.dcd")
        topfile = glob.glob(name + "_min_cent.pdb")[0]
    else:
        trajnames = glob.glob(name + "_traj_[1-9]*.dcd")
        topfile = glob.glob(name + "_noslv_min.pdb")[0]
    return topfile, trajnames

def histogram_runs(all_rg, rehist=False):

    hist_files = ["n.npy", "Pn.npy", "avg_Rg.dat", "bin_edges.npy", "mid_bin.npy"]
    all_hist_files_exist = np.all([ os.path.exists(savedir + "/" + x) for x in hist_files ])

    if all_hist_files_exist and not rehist: 
        Pn = np.load("Pn.npy")
        avgRg = float(np.loadtxt("avg_Rg.dat"))
        mid_bin = np.load("mid_bin.npy")

        if os.path.exists("dPn.npy"):
            dPn = np.load("dPn.npy")
        else:
            dPn = None
    else:
        # statistics with all runs and between runs.
        max_rg = np.max([ np.max(x) for x in all_rg ])
        min_rg = np.min([ np.min(x) for x in all_rg ])

        # calculate the distribution for each run
        bin_edges = np.linspace(min_rg, max_rg, 100)
        mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])
        n, _ = np.histogram(np.concatenate(all_rg), bins=bin_edges)
        Pn, _ = np.histogram(np.concatenate(all_rg), density=True, bins=bin_edges)

        avgRg = np.mean(np.concatenate(all_rg))

        np.save("n.npy", n)
        np.save("Pn.npy", Pn)
        np.save("bin_edges.npy", bin_edges)
        np.save("mid_bin.npy", mid_bin)
        np.savetxt("avg_Rg.dat", np.array([avgRg]))

        if len(all_rg) > 2: 
            dPn = np.std([ np.histogram(x, density=True, bins=bin_edges)[0] for x in all_rg ], axis=0)
            np.save("dPn.npy", dPn)
        else:
            dPn = None

    return mid_bin, Pn, dPn, avgRg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('--subdir', type=str, default=".", help='Name.')
    parser.add_argument('--recalc', action="store_true", help='Recalculate.')
    parser.add_argument('--rehist', action="store_true", help='Re-histogram.')
    parser.add_argument('--plot_trace', action="store_true", help="Plot time trace")
    parser.add_argument('--plot_hist', action="store_true", help="Plot histogram")
    parser.add_argument('--use_cent', action="store_true", help='Use centered traj.')
    args = parser.parse_args()

    name = args.name
    subdir = args.subdir
    recalc = args.recalc
    rehist = args.recalc
    plot_trace = args.plot_trace
    plot_hist = args.plot_hist
    use_cent = args.use_cent

    #name = "c25"
    savedir = "rg_dist"
    if use_cent:
        tname = "run_[1-9]*/" + name + "_traj_cent_[1-9]*.dcd"
    else:
        tname = "run_[1-9]*/" + name + "_traj_[1-9]*.dcd"

    cwd = os.getcwd()

    if len(glob.glob(subdir + "/*/T_*/" + tname)) > 0:
        # We are in directory above temps.
        Tpaths = glob.glob(subdir + "*/T_*")
    elif len(glob.glob(subdir + "/" + tname)) > 0:
        Tpaths = []
    elif len(glob.glob(subdir + "/T_*/" + tname)) > 0:
        Tpaths = glob.glob(subdir + "/T_*")
    else:
        raise ValueError("Couldn't find trajectories")

    hist_files = ["n.npy", "Pn.npy", "avg_Rg.dat", "bin_edges.npy", "mid_bin.npy"]
    if len(Tpaths) > 0:
        for i in range(len(Tpaths)):
            os.chdir(Tpaths[i])

            all_hist_files_exist = np.all([ os.path.exists(savedir + "/" + x) for x in hist_files ])
            all_rg_files_exist = rg_files_exist(name, use_cent=use_cent)

            if not all_hist_files_exist or not all_rg_files_exist or recalc:
                no_data, mid_bin, Pn, dPn, avgRg = histogram_runs(name, use_cent=use_cent, recalc=recalc)
            else:
                os.chdir(savedir)
                no_data = False
                Pn = np.load("Pn.npy")
                avgRg = float(np.loadtxt("avg_Rg.dat"))
                mid_bin = np.load("mid_bin.npy")

                if os.path.exists("dPn.npy"):
                    dPn = np.load("dPn.npy")
                else:
                    dPn = None
                os.chdir("..")
                
            if not no_data and plot:
                os.chdir(savedir)
                plt.figure()
                if dPn is not None:
                    plt.errorbar(mid_bin, Pn, yerr=dPn, lw=2)
                else:
                    plt.plot(mid_bin, Pn, lw=2)
                plt.xlabel("Radius gyration (nm)")
                plt.ylabel("Prob density")
                plt.title(Tpaths[i] + r" $\langle R_g \rangle = {:.2f}$".format(avgRg))
                plt.savefig("Rg_dist.pdf")
                plt.savefig("Rg_dist.png")
                os.chdir("..")
            os.chdir(cwd)
    else:
        all_rg = get_rg_for_all_runs(name, use_cent, recalc)

        if not os.path.exists(savedir):
            os.mkdir(savedir)
        os.chdir(savedir)

        if len(all_rg) > 0:
            if plot_trace:
                plot_vs_time(all_rg)

            mid_bin, Pn, dPn, avgRg = histogram_runs(all_rg, rehist=rehist)

            if plot_hist:
                plt.figure()
                if dPn is not None:
                    plt.errorbar(mid_bin, Pn, yerr=dPn, lw=2)
                else:
                    plt.plot(mid_bin, Pn, lw=2)
                plt.xlabel("Radius gyration (nm)")
                plt.ylabel("Prob density")
                plt.title(r"$\langle R_g \rangle = {:.2f}$".format(avgRg))
                plt.axvline(avgRg, ls='--', color='k')
                plt.savefig("Rg_dist.pdf")
                plt.savefig("Rg_dist.png")

                plt.figure()
                pmf = -np.log(Pn)
                pmf -= pmf.min()
                plt.plot(mid_bin, pmf, lw=2)
                plt.xlabel("Radius gyration (nm)")
                plt.ylabel("PMF (k$_B$T)")
                plt.savefig("Rg_pmf.pdf")
                plt.savefig("Rg_pmf.png")

        else:
            print("No rg to plot.")

        os.chdir("..")
