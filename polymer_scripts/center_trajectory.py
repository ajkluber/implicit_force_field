import os
import time
import glob
import argparse
import numpy as np

import mdtraj as md

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('--subdir', type=str, default=".", help='Subdirectory.')
    parser.add_argument('--rundirs', nargs="+", type=int, default=None, help='Subdirectory.')
    parser.add_argument('--nowait', action="store_true", help='Ignore if traj was recently written.')
    parser.add_argument('--recenter', action="store_true", help='Force calculation.')
    args = parser.parse_args()

    name = args.name
    subdir = args.subdir
    rundirs = args.rundirs
    nowait = args.nowait
    recenter = args.recenter

    if not rundirs is None:
        pass  
    
    if len(glob.glob(subdir + "/*/*/*/{}_traj_*.dcd".format(name))) > 0:
        trajpaths = glob.glob(subdir + "/*/*/run_*/{}_traj_*.dcd".format(name))
    elif len(glob.glob(subdir + "/run_*/{}_traj_*.dcd".format(name))) > 0:
        trajpaths = glob.glob(subdir + "/run_*/{}_traj_*.dcd".format(name))
    else:
        trajpaths = glob.glob(subdir + "/*/run_*/{}_traj_*.dcd".format(name))

    #print(str(trajpaths))

    cwd = os.getcwd()
    for i in range(len(trajpaths)):
        old_name = os.path.basename(trajpaths[i])
        #traj_idx = old_name.split(name + "_traj_")[1].split(".dcd")[0]
        traj_idx = old_name.split(".dcd")[0].split("_")[-1]
        new_name = name + "_traj_cent_" + traj_idx + ".dcd" 
        new_pdbname = name + "_min_cent.pdb" 

        os.chdir(os.path.dirname(trajpaths[i]))
        # only calc if trajectory if not currently running (i.e., hasn't been
        # modified in 5mins)
        last_change = np.abs(os.path.getmtime(old_name) - time.time())/60.
        if (not os.path.exists(new_name) or recenter) and ((last_change > 5) or not nowait):
            # get indices for polymer
            topfile = name + "_min.pdb"
            if not os.path.exists(topfile):
                topfile = name + "_min_1.pdb"
                if not os.path.exists(topfile):
                    raise ValueError("No pdb file present!")
                    
            pdb = md.load(topfile)
            ply_idxs = pdb.top.select("name PL") 

            print("centering:" + os.getcwd() + "/" + old_name + "  saving centered traj as:" + new_name)
            traj = md.load(old_name, top=pdb, atom_indices=ply_idxs)
            traj.center_coordinates()
            traj[0].save_pdb(new_pdbname)
            traj.save_dcd(new_name)
        os.chdir(cwd)
