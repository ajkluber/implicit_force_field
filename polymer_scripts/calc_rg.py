import os
import glob
import argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

import mdtraj as md

def get_rg(trajnames):
    Rg = []
    for chunk in md.iterload(trajnames[j], top=pdb, atom_indices=ply_idxs):
        rg = md.compute_rg(chunk)
        Rg.append(rg)
    Rg = np.concatenate(Rg)
    np.save("rg_{}.npy".format(traj_idx), Rg)
    return Rg

def calc_for_runs(name):
    runpaths = glob.glob("run_[1-9]*")
    # analyze all trajectories at one temperature
    all_rg = []
    for i in range(len(runpaths)):
        os.chdir(runpaths[i])
        trajnames = glob.glob(name + "_traj_cent_*.dcd")

        if len(trajnames) > 0:
            topfile = name + "_min_cent.pdb"
            pdb = md.load(topfile)
            ply_idxs = pdb.top.select("name PL") 

            rg_for_run = []
            print("calculating Rg for rundir:" + os.getcwd())
            for j in range(len(trajnames)):
                traj_idx = (trajnames[j]).split(".dcd")[0].split("_")[-1]
                Rg = []
                for chunk in md.iterload(trajnames[j], top=pdb, atom_indices=ply_idxs):
                    rg = md.compute_rg(chunk)
                    Rg.append(rg)
                Rg = np.concatenate(Rg)
                print("  " + "rg_{}.npy".format(traj_idx))
                np.save("rg_{}.npy".format(traj_idx), Rg)
                rg_for_run.append(Rg)
            all_rg.append(np.concatenate(rg_for_run))
        os.chdir("..")

    if len(all_rg) > 0:
        no_data = False
        # statistics with all runs and between runs.
        max_rg = np.max([ np.max(x) for x in all_rg ])
        min_rg = np.min([ np.min(x) for x in all_rg ])

        # calculate the distribution for each run
        bin_edges = np.linspace(min_rg, max_rg, 100)
        mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])
        n, _ = np.histogram(np.concatenate(all_rg), bins=bin_edges)
        Pn, _ = np.histogram(np.concatenate(all_rg), density=True, bins=bin_edges)

        avgRg = np.mean(np.concatenate(all_rg))

        if not os.path.exists(savedir):
            os.mkdir(savedir)
        os.chdir(savedir)
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
    else:
        no_data = True
        mid_bin, Pn, dPn, avgRg = None, None, None, None
    return no_data, mid_bin, Pn, dPn, avgRg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('--subdir', type=str, default=".", help='Name.')
    parser.add_argument('--recalc', action="store_true", help='Recalculate.')
    parser.add_argument('--plot', action="store_true")
    args = parser.parse_args()

    name = args.name
    subdir = args.subdir
    recalc = args.recalc
    plot = args.plot

    #name = "c25"
    savedir = "rg_dist"
    tname = "run_[1-9]*/" + name + "_traj_*.dcd"

    #os.chdir(subdir)
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

    files = ["n.npy", "Pn.npy", "avg_Rg.dat", "bin_edges.npy", "mid_bin.npy"]
    if len(Tpaths) > 0:
        for i in range(len(Tpaths)):
            #print(os.getcwd())
            os.chdir(Tpaths[i])
            all_files_exist = np.all([ os.path.exists(savedir + "/" + x) for x in files ])
            if not all_files_exist or recalc:
                no_data, mid_bin, Pn, dPn, avgRg = calc_for_runs(name)
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
        all_files_exist = np.all([ os.path.exists(savedir + "/" + x) for x in files ])
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        if not all_files_exist or recalc:
            no_data, mid_bin, Pn, dPn, avgRg = calc_for_runs(name)
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
            os.chdir("..")
