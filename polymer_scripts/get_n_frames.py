
import implicit_force_field.util as util

import os
import time
import glob
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('subdir', type=str, help='Subdirectory.')
    parser.add_argument('--nowait', action="store_true", help='Ignore if traj was recently written.')
    parser.add_argument('--recenter', action="store_true", help='Force calculation.')
    args = parser.parse_args()

    name = args.name
    subdir = args.subdir

    os.chdir(subdir)
    cwd = os.getcwd()

    if len(glob.glob(subdir + "*/T_*")) > 0:
        # We are in directory above temps.
        Tpaths = glob.glob("*/T_*")
    else:
        Tpaths = glob.glob("T_*")

    topfile = name + "_min_cent.pdb"

    for i in range(len(Tpaths)):
        os.chdir(Tpaths[i])

        runpaths = [ x.split("/")[0] for x in glob.glob("run_*/{}_min_cent.pdb".format(name)) ]
        #runpaths.sort()
        print "T = {}".format(Tpaths[i])

        #status_str = ""
        for j in range(len(runpaths)):
            os.chdir(runpaths[j])
            trajnames = glob.glob(name + "_traj_cent_*.dcd")
            #print os.getcwd(), len(trajnames)

            status_str = "{:<6s} ".format(runpaths[j])
            status_str += " ".join([ "{:^9s}".format((x).split(".dcd")[0].split("_")[-1]) for x in trajnames ])
            status_str += "  TOTAL\n"
            status_str += 7*" "
            
            Ntot = 0
            for n in range(len(trajnames)):
                N, _ = util.get_n_frames(trajnames[n], topfile)
                Ntot += N
                status_str += "{:.2e}  ".format(N)
            status_str += "  {:.2e}".format(Ntot) + "\n"

            print status_str

            os.chdir("..")
        os.chdir("..")
        #print status_str
        break

