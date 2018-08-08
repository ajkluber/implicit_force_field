import os
import glob
import argparse
import numpy as np

import mdtraj as md

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('subdir', type=str, help='Name.')
    args = parser.parse_args()

    subdir = args.subdir

    name = "c25"
    
    trajpaths = glob.glob(subdir + "/*/*/*/{}_traj_*.dcd".format(name))

    cwd = os.getcwd()
    for i in range(len(trajpaths)):
        old_name = os.path.basename(trajpaths[i])
        #traj_idx = old_name.split(name + "_traj_")[1].split(".dcd")[0]
        traj_idx = old_name.split(".dcd")[0].split("_")[-1]
        new_name = name + "_traj_cent_" + traj_idx + ".dcd" 
        new_pdbname = name + "_min_cent.pdb" 

        os.chdir(os.path.dirname(trajpaths[i]))
        if not os.path.exists(new_name):
            # get indices for polymer
            topfile = name + "_min.pdb"
            pdb = md.load(topfile)
            ply_idxs = pdb.top.select("name PL") 

            print "centering:", old_name, "  saving centered traj as:", new_name
            traj = md.load(old_name, top=pdb, atom_indices=ply_idxs)
            traj.center_coordinates()
            traj[0].save_pdb(new_pdbname)
            traj.save_dcd(new_name)
        os.chdir(cwd)
