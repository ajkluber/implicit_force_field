import os
import glob
import pickle
import argparse
import numpy as np 


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
    from pyemma.coordinates.data.featurization.misc import CustomFeature


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
    feat.add_distances(pair_idxs)
    #feature_info = {'pairs':pair_idxs, 'r0':r0, 'widths':widths, 'dim':len(pair_idxs)}

    reader = coor.source(trajnames, features=feat)

    keep_dims = 20
    #lagtimes = [1, 5, 10, 100, 500, 1000]
    lagtimes = [1, 2, 5, 8, 10, 20, 50]
    ti = []
    cvar = []
    for n in range(len(lagtimes)):
        tica = coor.tica(lag=lagtimes[n], stride=1, dim=keep_dims)
        coor.pipeline([reader, tica])

        ti.append(tica.timescales)
        cvar.append(tica.cumvar)

    ti = np.array(ti)
   
    plt.figure()
    for i in range(10):
        #imp_ti = -np.log(ti[:,i])/lagtime[
        plt.plot(lagtimes, ti[:,i])

    plt.fill_between(lagtimes, lagtimes, color='k', facecolor='gray', lw=2)
    plt.xlabel("Lagtime")
    plt.ylabel("Imp Timescale")
    plt.title(name + " TICA")
    plt.savefig("its.pdf")
    
    idxs = np.arange(1, keep_dims + 1)

    plt.figure()
    for i in range(len(lagtimes)):
        plt.plot(idxs, ti[i][:keep_dims], 'o', label=str(lagtimes[i]))

    plt.legend(loc=1)
    plt.xlim(0, idxs[-1])
    plt.xticks(idxs)
    plt.xlabel("Index")
    plt.ylabel("Imp Timescale")
    plt.savefig("ti_vs_index.pdf")

    plt.figure()
    for i in range(len(lagtimes)):
        plt.plot(idxs, cvar[i][:keep_dims], 'o', label=str(lagtimes[i]))
    plt.legend(loc=2)
    plt.xlim(0, idxs[-1])
    plt.xticks(idxs)
    plt.xlabel("Index")
    plt.ylabel("Cum. Var.")
    plt.savefig("cumvar_vs_index.pdf")


    tica = coor.tica(lag=10, stride=1, dim=10)
    cluster = coor.cluster_kmeans(k=500)
    coor.pipeline([reader, tica, cluster])
    M = msm.estimate_markov_model(cluster.dtrajs, 1)
    
    msm_lags = [1, 10, 20, 50, 100, 200]
    its = msm.its(cluster.dtrajs, lags=msm_lags)
    plt.figure()
    mplt.plot_implied_timescales(its, ylog=False)
    plt.savefig("msm_its.pdf")



    plt.figure()
    plt.plot(np.arange(1,21), M.timescales()[:20], 'o')
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    plt.savefig("msm_ti.pdf")
