import os
import glob
import argparse
import numpy as np
#import matplotlib.pyplot as plt

import mdtraj as md

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('subdir', type=str, help='Name.')
    parser.add_argument('--recalc', action="store_true", help='Recalculate.')
    args = parser.parse_args()

    name = args.name
    subdir = args.subdir
    recalc = args.recalc

    savedir = "q_dist"

    r0 = 0.4
    w = 0.1
    contact = lambda r: 0.5*(np.tanh(-(r - r0)/w) + 1)

    os.chdir(subdir)
    cwd = os.getcwd()

    if len(glob.glob("*/T_*")) > 0:
        # We are in directory above temps.
        Tpaths = glob.glob("*/T_*")
    else:
        Tpaths = glob.glob("T_*")

    for i in range(len(Tpaths)):
        #print os.getcwd()
        os.chdir(Tpaths[i])
        runpaths = glob.glob("run_*")

        files = ["n.npy", "Pn.npy", "avg_Q.dat", "bin_edges.npy", "mid_bin.npy"]
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

                    q_for_run = []
                    print "calculating Q for rundir:", os.getcwd()
                    for j in range(len(trajnames)):
                        traj_idx = (trajnames[j]).split(".dcd")[0].split("_")[-1]

                        Q = []
                        for chunk in md.iterload(trajnames[j], top=pdb, atom_indices=ply_idxs):
                            q = np.sum(contact(md.compute_distances(chunk, pairs)), axis=1)
                            Q.append(q)
                        Q = np.concatenate(Q)
                        print "  ", "q_{}.npy".format(traj_idx)
                        np.save("q_{}.npy".format(traj_idx), Q)

                        if traj_idx == 1:
                            q_for_run.append(Q[200:])
                        else:
                            q_for_run.append(Q)
                    all_rg.append(np.concatenate(q_for_run))
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

                avgQ = np.mean(np.concatenate(all_rg))

                if not os.path.exists(savedir):
                    os.mkdir(savedir)
                os.chdir(savedir)
                np.save("n.npy", n)
                np.save("Pn.npy", Pn)
                np.save("bin_edges.npy", bin_edges)
                np.save("mid_bin.npy", mid_bin)
                np.savetxt("avg_Q.dat", np.array([avgQ]))

                if len(all_rg) > 2: 
                    dPn = np.std([ np.histogram(x, density=True, bins=bin_edges)[0] for x in all_rg ], axis=0)
                    np.save("dPn.npy", dPn)

        os.chdir(cwd)
