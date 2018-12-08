import os
import glob
import argparse
import numpy as np

import mdtraj as md

def get_rg(trajnames):
    Rg = []
    for chunk in md.iterload(trajnames[j], top=pdb, atom_indices=ply_idxs):
        rg = md.compute_rg(chunk)
        Rg.append(rg)
    Rg = np.concatenate(Rg)
    np.save("rg_{}.npy".format(traj_idx), Rg)
    return Rg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('subdir', type=str, help='Name.')
    parser.add_argument('--recalc', action="store_true", help='Recalculate.')
    args = parser.parse_args()

    name = args.name
    subdir = args.subdir
    recalc = args.recalc

    #name = "c25"

    savedir = "rg_dist"

    os.chdir(subdir)
    cwd = os.getcwd()

    if len(glob.glob("*/T_*")) > 0:
        # We are in directory above temps.
        Tpaths = glob.glob("*/T_*")
    else:
        Tpaths = glob.glob("T_*")

    for i in range(len(Tpaths)):
        #print(os.getcwd())
        os.chdir(Tpaths[i])
        runpaths = glob.glob("run_[1-9]*")

        files = ["n.npy", "Pn.npy", "avg_Rg.dat", "bin_edges.npy", "mid_bin.npy"]
        all_files_exist = np.all([ os.path.exists(savedir + "/" + x) for x in files ])

        if not all_files_exist or recalc:
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

        os.chdir(cwd)
