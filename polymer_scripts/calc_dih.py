import os
import glob
import argparse
import numpy as np

import mdtraj as md

def calc_for_dih22():
    pdb = md.load("c25_min_1.pdb")
    ply_idxs = pdb.topology.select("resname PLY")

    subdir = "dih_dists" 
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    
    dih_idxs = np.array([21,22,23,24])

    bin_edges = np.linspace(-np.pi, np.pi, 100)
    mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

    all_hist = np.zeros(len(mid_bin), float)

    n_frames = 0
    # for each dihedral. calculate the distribution
    for chunk in md.iterload("c25_traj_1.dcd", top=pdb, atom_indices=ply_idxs):
        phi = md.compute_dihedrals(chunk, np.array([[0,1,2,3]]))[:,0]
        all_hist += np.histogram(phi, bins=bin_edges)[0]

    np.save("{}/dih_22.npy".format(subdir), all_hist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('subdir', type=str, help='Name.')
    parser.add_argument('--recalc', action="store_true", help='Recalculate.')
    args = parser.parse_args()

    name = args.name
    subdir = args.subdir
    recalc = args.recalc

    savedir = "dih_dists" 

    os.chdir(subdir)
    cwd = os.getcwd()

    if len(glob.glob("*/T_*")) > 0:
        # We are in directory above temps.
        Tpaths = glob.glob("*/T_*")
        topfile = glob.glob("*/T_*/run_*/" + name + "_min_cent.pdb")[0]
    else:
        Tpaths = glob.glob("T_*")
        topfile = glob.glob("T_*/run_*/" + name + "_min_cent.pdb")[0]

    bin_edges = np.linspace(-np.pi, np.pi, 100)
    mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

    pdb = md.load(topfile)
    ply_idxs = pdb.top.select("resname PLY") 

    dih_idxs = []
    for i in range(len(ply_idxs) - 3):
        idx = ply_idxs[i]
        dih_idxs.append([idx, idx + 1, idx + 2, idx + 3])
    dih_idxs = np.array(dih_idxs)
    n_ang = len(dih_idxs)

    # Indicator function of global helical content using Von Mises function.
    # Use only internal dihedrals because ends are floppy.
    phi_L = (5./8)*np.pi
    phi_R = -(5./8)*np.pi
    kphi = 5.

    helix_fraction = lambda data, phi0, kphi: np.sum(np.exp(kphi*np.cos(data[:,4:-4] - phi0))/np.exp(kphi), axis=1)

    my_hist = lambda data: np.histogram(data, bins=bin_edges)[0]

    for i in range(len(Tpaths)):
        print " For:",Tpaths[i]
        os.chdir(Tpaths[i])
        runpaths = glob.glob("run_[1-9]*")

        if not os.path.exists(savedir):
            os.mkdir(savedir)

        files = ["n.npy", "Pn.npy", "avg_dih.npy", "bin_edges.npy", "mid_bin.npy"]
        all_files_exist = np.all([ os.path.exists(savedir + "/" + x) for x in files ])

        if not all_files_exist or recalc:
            # analyze all trajectories at one temperature
            n_frames = 0.
            all_hist = np.zeros((n_ang, len(mid_bin)), float)
            for i in range(len(runpaths)):
                os.chdir(runpaths[i])
                trajnames = glob.glob(name + "_traj_cent_*.dcd")

                if len(trajnames) > 0:
                    print "calculating Rg for rundir:", os.getcwd()
                    for j in range(len(trajnames)):
                        traj_idx = (trajnames[j]).split(".dcd")[0].split("_")[-1]
                        run_L = []
                        run_R = []
                        for chunk in md.iterload(trajnames[j], top=pdb, atom_indices=ply_idxs):
                            phi = md.compute_dihedrals(chunk, dih_idxs)
                            run_L.append(helix_fraction(phi, phi_L, kphi))
                            run_R.append(helix_fraction(phi, phi_R, kphi))
                            all_hist += np.array(map(my_hist, phi.T))
                            n_frames += chunk.n_frames
                        np.save("helix_L_{}.npy".format(traj_idx), np.concatenate(run_L))
                        np.save("helix_R_{}.npy".format(traj_idx), np.concatenate(run_R))
                os.chdir("..")
            np.save(savedir + "/dih_idxs.npy", dih_idxs)
            np.save(savedir + "/dih_dists.npy", all_hist)
            np.save(savedir + "/bin_edges.npy", bin_edges)
            np.savetxt(savedir + "/n_frames.dat", np.array([n_frames]))

            #if len(all_rg) > 0:
            #    # statistics with all runs and between runs.
            #    max_rg = np.max([ np.max(x) for x in all_rg ])
            #    min_rg = np.min([ np.min(x) for x in all_rg ])

            #    # calculate the distribution for each run
            #    bin_edges = np.linspace(min_rg, max_rg, 100)
            #    mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])
            #    n, _ = np.histogram(np.concatenate(all_rg), bins=bin_edges)
            #    Pn, _ = np.histogram(np.concatenate(all_rg), density=True, bins=bin_edges)

            #    avgRg = np.mean(np.concatenate(all_rg))

            #    if not os.path.exists(savedir):
            #        os.mkdir(savedir)
            #    os.chdir(savedir)
            #    np.save("n.npy", n)
            #    np.save("Pn.npy", Pn)
            #    np.save("bin_edges.npy", bin_edges)
            #    np.save("mid_bin.npy", mid_bin)
            #    np.savetxt("avg_Rg.dat", np.array([avgRg]))

            #    if len(all_rg) > 2: 
            #        dPn = np.std([ np.histogram(x, density=True, bins=bin_edges)[0] for x in all_rg ], axis=0)
            #        np.save("dPn.npy", dPn)

        os.chdir(cwd)

