import os
import sys
import glob
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt

def tanh_contact(traj, pairs, r0, widths):
    r = md.compute_distances(traj, pairs)
    return 0.5*(np.tanh((r0 - r)/widths) + 1)

def save_markov_state_models(T, models):

    msm_info = {}
    msm_info["temperature"] = T
    msm_info["lagtimes"] = [ x.lagtime for x in models ]

    for i in range(len(models)):
        lagtime = models[i].lagtime
        msm_info[str(lagtime)] = models[i].transition_matrix

    with open("msm.pkl", "wb") as fhandle:
        pickle.dump(msm_info, fhandle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--lagtime", type=int, default=50)
    parser.add_argument("--keep_dims", type=int, default=5)
    parser.add_argument("--n_clusters", type=int, default=500)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--use_dihedrals", action="store_true")
    parser.add_argument("--use_distances", action="store_true")
    parser.add_argument("--use_inv_distances", action="store_true")
    parser.add_argument("--use_rg", action="store_true")
    parser.add_argument("--resave_tic", action="store_true")
    parser.add_argument("--use_saved_tics", action="store_true")
    args = parser.parse_args()

    name = args.name
    lagtime = args.lagtime
    keep_dims = args.keep_dims
    n_clusters = args.n_clusters
    stride = args.stride
    use_dihedrals = args.use_dihedrals
    use_distances = args.use_distances
    use_inv_distances = args.use_inv_distances
    use_rg = args.use_rg
    resave_tic = args.resave_tic
    use_saved_tics = args.use_saved_tics

    feature_set = []
    if use_dihedrals:
        feature_set.append("dih")
    if use_distances:
        feature_set.append("dists")
    if use_inv_distances:
        feature_set.append("invdists")
    if use_rg:
        feature_set.append("rg")

    f_str = "_".join(feature_set)
    msm_savedir = "msm_" + f_str

    # save the command that was run to make output
    with open(msm_savedir + "/cmd.txt", "a") as fout:
        fout.write("python " + " ".join(sys.argv) + "\n")

    # get trajectory names
    topfile = glob.glob("run_*/" + name + "_min_cent.pdb")[0]

    trajnames = glob.glob("run_*/" + name + "_traj_cent_*.dcd") 
    traj_idxs = []
    for i in range(len(trajnames)):
        tname = trajnames[i]
        idx1 = (os.path.dirname(tname)).split("_")[-1]
        idx2 = (os.path.basename(tname)).split(".dcd")[0].split("_")[-1]
        traj_idxs.append([idx1, idx2])

    cluster = coor.cluster_kmeans(k=n_clusters)

    if use_saved_tics:
        Y = []
        for n in range(len(traj_idxs)):
            idx1, idx2 = traj_idxs[n]
            tic_traj_temp = []
            for i in range(keep_dims):
                # save TIC with indices of corresponding traj
                tic_saveas = msm_savedir + "/run_{}_{}_TIC_{}.npy".format(idx1, idx2, i+1)
                tic_traj_temp.append(np.load(tic_saveas))
            Y.append(np.array(tic_traj_temp).T)

        reader = coor.source(Y)
        cluster_pipe = [reader, cluster] 
    else:
        # get indices for dihedral angles and pairwise distances
        feat = coor.featurizer(topfile)
        ply_idxs = feat.topology.select("name PL")

        dih_idxs = np.array([[ply_idxs[i], ply_idxs[i + 1], ply_idxs[i + 2], ply_idxs[i + 3]] for i in range(len(ply_idxs) - 4) ])
        pair_idxs = []
        for i in range(len(ply_idxs) - 1):
            for j in range(i + 4, len(ply_idxs)):
                pair_idxs.append([ply_idxs[i], ply_idxs[j]])
        pair_idxs = np.array(pair_idxs)

        all_pair_idxs = []
        for i in range(len(ply_idxs)):
            this_residue = []
            for j in range(len(ply_idxs)):
                if np.abs(i - j) > 3:
                    if i < j:
                        this_residue.append([ply_idxs[i], ply_idxs[j]])
                    elif j < i:
                        this_residue.append([ply_idxs[j], ply_idxs[i]])
                    else:
                        pass
            all_pair_idxs.append(this_residue)
        all_pair_idxs = np.array(all_pair_idxs)

        # add dihedrals
        if "dih" in feature_set:
            feat.add_dihedrals(dih_idxs, cossin=True)
        if "tanh" in feature_set:
            feat.add_custom_feature(CustomFeature(tanh_contact, len(pair_idxs), fun_args=(pair_idxs, r0, widths)))
        if "dists" in feature_set:
            feat.add_distances(pair_idxs)
        if "invdists" in feature_set:
            feat.add_inverse_distances(pair_idxs)
        if "rho" in feature_set:
            feat.add_custom_feature(CustomFeature(local_density_feature, len(all_pair_idxs), fun_args=(all_pair_idxs, r0, widths)))
        if "rg" in feature_set:
            feat.add_custom_feature(CustomFeature(rg_feature, 1))

        reader = coor.source(trajnames, features=feat)
        transform = coor.tica(lag=lagtime, stride=stride)
        cluster_pipe = [reader, transform, cluster] 
        
    pipeline = coor.pipeline(cluster_pipe)
    dtrajs = cluster.dtrajs
    #lags = [1,2,5,10,20,50,100,200,300,400,500,600,700,800,900,1000]
    lags = [10,25,50,100,200,500,1000]
    its = msm.its(dtrajs, lags=lags)

    #if savemsm:
    #    save_markov_state_models(T, its.models)

    # save name should have n_clusters
    saveas = "msm_its_vs_lag_{}".format(n_clusters)
    mplt.plot_implied_timescales(its, ylog=False)
    #plt.title("T = " + str(T))
    plt.savefig(msm_savedir + "/" + saveas + ".pdf")
    plt.savefig(msm_savedir + "/" + saveas + ".png")

    plt.figure()
    mplt.plot_implied_timescales(its)
    #plt.title("T = " + str(T))
    plt.savefig(msm_savedir + "/" + saveas + "_ylog.pdf")
    plt.savefig(msm_savedir + "/" + saveas + "_ylog.png")


    raise SystemExit

    dt = 0.2 # ps

    stride = 10
    tica_lag = 10
    n_clusters = 100
    topfile = name + "_min_cent_1.pdb"
    trajfiles = ["c25_traj_1.dcd"]
    recluster = True

    pdb = md.load("c25_min_1.pdb")
    ply_idxs = pdb.topology.select("resname PLY")

    r0 = 0.3
    width = 0.1

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

