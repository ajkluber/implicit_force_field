import os
import glob
import pickle
import argparse
import numpy as np 

from pyemma.coordinates.data.featurization.misc import CustomFeature

import mdtraj as md

def tanh_contact(traj, pairs, r0, widths):
    r = md.compute_distances(traj, pairs)
    return 0.5*(np.tanh((r0 - r)/widths) + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    #parser.add_argument("subdir", type=str)
    args = parser.parse_args()

    name = args.name
    #subidr = args.subdir

    display = False

    if display:
        import pyemma.plots as mplt
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pyemma.plots as mplt

    import pyemma.coordinates as coor
    import pyemma.msm as msm


    # get trajectory names
    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd")
    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]

    # get indices for dihedral angles and pairwise distances
    feat = coor.featurizer(topfile)
    ply_idxs = feat.topology.select("name PL")

    dih_idxs = np.array([[ply_idxs[i], ply_idxs[i + 1], ply_idxs[i + 2], ply_idxs[i + 3]] for i in range(len(ply_idxs) - 4) ])
    pair_idxs = []
    for i in range(len(ply_idxs) - 1):
        for j in range(i + 3, len(ply_idxs)):
            pair_idxs.append([ply_idxs[i], ply_idxs[j]])
    pair_idxs = np.array(pair_idxs)

    r0 = 0.4
    widths = 0.1

    # add dihedrals
    #feat.add_custom_feature(CustomFeature(tanh_contact, len(pair_idxs), fun_args=(pair_idxs, r0, widths)))
    feat.add_dihedrals(dih_idxs, cossin=True)
    feat.add_distances(pair_idxs, cossin=True)
    #feature_info = {'pairs':pair_idxs, 'r0':r0, 'widths':widths, 'dim':len(pair_idxs)}

    reader = coor.source(trajnames, features=feat)

    lagtimes = [1, 5, 10, 100, 500, 1000]
    for n in range(len(lagtimes)):
        tica = coor.tica(lag=lagtimes[n], stride=1)
        coor.pipeline([reader, tica])

