import os
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdtraj as md

import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt

def tanh_contact(traj, pairs, r0, widths):
    r = md.compute_distances(traj, pairs)
    return 0.5*(np.tanh((r0 - r)/widths) + 1)


if __name__ == "__main__":
    dt = 0.2 # ps

    stride = 10
    tica_lag = 10
    n_clusters = 100
    topfile = "c25_min_1.pdb"
    trajfiles = ["c25_traj_1.dcd"]
    recluster = True

    pdb = md.load("c25_min_1.pdb")
    ply_idxs = pdb.topology.select("resname PLY")

    # add features

    pair_idxs = []
    for i in range(len(ply_idxs) - 1):
        for j in range(i + 3, len(ply_idxs)):
            pair_idxs.append([ply_idxs[i], ply_idxs[j]])
    pair_idxs = np.array(pair_idxs)

    dih_idxs = []
    for i in range(len(ply_idxs) - 4):
        idx = ply_idxs[i]
        dih_idxs.append([idx, idx + 1, idx + 2, idx + 3])
    dih_idxs = np.array(dih_idxs)

    r0 = 0.3
    width = 0.1

    feat = coor.featurizer(topfile)
    #feat.add_custom_feature(CustomFeature(tanh_contact, pair_idxs, r0, width, dim=len(pair_idxs)))
    feat.add_distances(pair_idxs)
    feat.add_dihedrals(dih_idxs)

    if not os.path.exists("msm"):
        os.mkdir("msm")

    if (not os.path.exists("msm/dtrajs.pkl")) or recluster:
        # cluster if necessary
        reader = coor.source(trajfiles, feat)
        transform = coor.tica(lag=tica_lag, stride=stride)
        cluster = coor.cluster_kmeans(k=n_clusters)
        pipeline = coor.pipeline([reader, transform, cluster])
        dtrajs = cluster.dtrajs

        os.chdir("msm")
        dirs = [ os.path.basename(os.path.dirname(x)) for x in trajfiles ]

        if not dontsavemsm:
            np.save("dtrajs.npy", dtrajs[0])
            #dtraj_info = { dirs[x]:dtrajs[x] for x in range(len(dirs)) }
            #dtraj_info["dirs"] = dirs
            #with open("dtrajs.pkl", 'wb') as fhandle:
            #    pickle.dump(dtraj_info, fhandle)
    else:
        os.chdir("msm")
        dtrajs = [np.load("dtrajs.npy")]
        #with open("dtrajs.pkl", 'rb') as fhandle:
        #    dtraj_pkl = pickle.load(fhandle)
        #    dirs = dtraj_pkl["dirs"]
        #    dtrajs = [ dtraj_pkl[x] for x in dirs ]

    # estimate MSM's at different lagtimes
    lags = [1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,1000]
    its = msm.its(dtrajs, lags=lags)

    #if not dontsavemsm:
    #    util.save_markov_state_models(T, its.models)

    mplt.plot_implied_timescales(its, ylog=False)
    #plt.title("T = " + str(T))
    plt.savefig("its_tanh_cont_features.pdf")
    plt.savefig("its_tanh_cont_features.png")
    os.chdir("..")

