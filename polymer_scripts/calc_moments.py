import os
import glob
import argparse
import numpy as np

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

    os.chdir(subdir)
    cwd = os.getcwd()

    #savedir = "dih_dists"

    if len(glob.glob("*/T_*")) > 0:
        # We are in directory above temps.
        Tpaths = glob.glob("*/T_*")
        topfile = glob.glob("*/T_*/run_*/" + name + "_min_cent.pdb")[0]
    else:
        Tpaths = glob.glob("T_*")
        topfile = glob.glob("T_*/run_*/" + name + "_min_cent.pdb")[0]

    pdb = md.load(topfile)
    ply_idxs = pdb.top.select("resname PLY") 

    for i in range(len(Tpaths)):
        print(" For:" + Tpaths[i])
        os.chdir(Tpaths[i])
        runpaths = glob.glob("run_[1-9]*")

        # analyze all trajectories at one temperature
        for i in range(len(runpaths)):
            os.chdir(runpaths[i])
            trajnames = glob.glob(name + "_traj_cent_*.dcd")

            if len(trajnames) > 0:
                print("calculating Rg for rundir:" + os.getcwd())
                for j in range(len(trajnames)):
                    traj_idx = (trajnames[j]).split(".dcd")[0].split("_")[-1]
                    gyr_eig = []
                    for chunk in md.iterload(trajnames[j], top=pdb, atom_indices=ply_idxs):
                        gyr_eig.append(np.linalg.eigvalsh(np.einsum("tim,tin->tmn", chunk.xyz, chunk.xyz)))
                    np.save("gyr_moments_{}.npy".format(traj_idx), np.concatenate(gyr_eig))
            os.chdir("..")
        os.chdir(cwd)

