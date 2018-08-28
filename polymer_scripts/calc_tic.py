import os
import glob
import pickle
import argparse
import numpy as np 


import mdtraj as md

def tanh_contact(traj, pairs, r0, widths):
    r = md.compute_distances(traj, pairs)
    return 0.5*(np.tanh((r0 - r)/widths) + 1)

def rg_feature(traj):
    return md.compute_rg(traj).astype(np.float32).reshape(-1, 1)

def local_density_feature(traj, pairs, r0, widths):

    rho_i = np.zeros((traj.n_frames, traj.n_atoms), np.float32)
    for i in range(len(pairs)):
        r = md.compute_distances(traj, pairs[i])
        rho_i[:,i] = np.sum(0.5*(np.tanh((r0 - r)/widths) + 1))
    #r = md.compute_distances(traj, pairs)
    #rho_i = np.sum(0.5*(np.tanh((r0 - r)/widths) + 1), axis=1).astype(np.float32).reshape(-1,1)
    return rho_i

def tica_stuff():

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

    #feature_set = ["dih", "invdists"]
    feature_set = ["dih", "tanh", "rho"]
    #feature_set = ["dih", "rho"]
    #feature_set = ["dih", "invdists", "rg"]
    #feature_set = ["dih", "dists"]
    #feature_set = ["dih", "dists", "rg"]
    msm_savedir = "msm_" + "_".join(feature_set)

    if not os.path.exists(msm_savedir):
        os.mkdir(msm_savedir)

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

    all_pair_idxs = []
    for i in range(len(ply_idxs)):
        this_residue = []
        for j in range(len(ply_idxs)):
            if i < j:
                this_residue.append([ply_idxs[i], ply_idxs[j]])
            elif j < i:
                this_residue.append([ply_idxs[j], ply_idxs[i]])
            else:
                pass
        all_pair_idxs.append(this_residue)
    all_pair_idxs = np.array(all_pair_idxs)

    r0 = 0.4
    widths = 0.1

    # add dihedrals
    if "dih" in feature_set:
        feat.add_dihedrals(dih_idxs, cossin=True)
    if "tanh" in feature_set:
        feat.add_custom_feature(CustomFeature(tanh_contact, len(pair_idxs), fun_args=(pair_idxs, r0, widths)))
    if "dists" in feature_set:
        feat.add_distances(pair_idxs)
    if "invdists" in feature_set:
        feat.add_inverse_distances(pair_idxs)
    if "rho":
        feat.add_custom_feature(CustomFeature(local_density_feature, len(all_pair_idxs), fun_args=(all_pair_idxs, r0, widths)))
    if "rg" in feature_set:
        feat.add_custom_feature(CustomFeature(rg_feature, 1))

    #feature_info = {'pairs':pair_idxs, 'r0':r0, 'widths':widths, 'dim':len(pair_idxs)}

    reader = coor.source(trajnames, features=feat)

    tica_lag = 10
    keep_dims = 10
    n_clusters = 100
    msm_lags = [1, 10, 20, 50, 100, 200]

    tica = coor.tica(lag=tica_lag, stride=1, dim=keep_dims)
    cluster = coor.cluster_kmeans(k=n_clusters)
    coor.pipeline([reader, tica, cluster])
    its = msm.its(cluster.dtrajs, lags=msm_lags)

    plt.figure()
    mplt.plot_implied_timescales(its, ylog=False)
    plt.title(msm_savedir)
    plt.savefig(msm_savedir + "/its_vs_lag.pdf")

    #plt.figure()
    #plt.plot(np.arange(1,21), M.timescales()[:20], 'o')
    #ymin, ymax = plt.ylim()
    #plt.ylim(0, ymax)
    #plt.savefig("msm_ti.pdf")
