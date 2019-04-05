import os
import glob
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

import simtk.unit as unit
import simtk.openmm.app as app

import mdtraj as md


import implicit_force_field.polymer_scripts.util as util

def sample_starting_configurations(n_samples, bin_file="psi1_mid_bin.npy",
        hist_file="psi1_n.npy", plot=False, sample_by_dist=False):
    """Sample starting configuations using the distribution"""

    # the cumulative distribution
    mid_bin = np.load(bin_file)
    hist_vals = np.load(hist_file)
    dx = mid_bin[1] - mid_bin[0]
    bin_edges = np.array([ (x - dx) for x in mid_bin ] + [ mid_bin[-1] + dx ])

    Prob = hist_vals/float(np.sum(hist_vals))
    Cumul = np.concatenate([np.array([0]), np.cumsum(Prob)])

    #if plot:
    #    plt.figure()
    #    plt.plot(bin_edges, Cumul)
    #    plt.xlabel("$\psi_2$")
    #    plt.ylabel("Cumulative dist fun")
    #    plt.savefig("cumulative_psi1.pdf")
    #    plt.savefig("cumulative_psi1.png")

    ticnames = glob.glob("run_*_TIC_1.npy")

    n_frames_traj = []
    traj_idxs = []
    tic_trajs = []
    for i in range(len(ticnames)):
        tname = ticnames[i]
        idx1 = tname.split("_")[1] 
        idx2 = tname.split("_")[2] 

        traj_idxs.append([idx1, idx2])

        temp = np.load(tname)
        n_frames_traj.append(temp.shape[0])
        tic_trajs.append(temp)

    # cumulative function of weights
    weights = [ n_frames_traj[i]/float(np.sum(n_frames_traj)) for i in range(len(n_frames_traj)) ]
    C_traj_weights = np.concatenate([ np.array([0]), np.cumsum(weights) ])

    n_samples = 100
    samp_vals = []
    samp_traj_idxs = []
    for i in range(n_samples):
        q1 = np.random.uniform()

        if sample_by_dist:
            for n in range(len(Cumul) - 1):
                # use cumulative distribution function to 
                # sample distribution
                if (q1 > Cumul[n]) and (q1 <= Cumul[n + 1]):
                    pick_psi = mid_bin[n]
                    break
        else:
            pick_psi = mid_bin[0] + q1*(mid_bin[-1] - mid_bin[0])

        samp_vals.append(pick_psi)

        if i == (n_samples - 1):
            print("assigning: {}/{}".format(i + 1, n_samples))
        else:
            print("assigning: {}/{}".format(i + 1, n_samples), end="\r")

        # assign a frame with this value of psi
        found_frame = False
        while not found_frame:
            q2 = np.random.uniform()
            for n in range(len(C_traj_weights) - 1):
                if (q2 > C_traj_weights[n]) and (q2 <= C_traj_weights[n + 1]):
                    pick_traj = n

                    pick_frame = np.argmin((tic_trajs[pick_traj] - pick_psi)**2)
                    err = np.abs(tic_trajs[pick_traj][pick_frame] - pick_psi)/pick_psi
                    if err <= 0.05:
                        samp_traj_idxs.append([pick_traj, pick_frame])
                        found_frame = True
                    else:
                        break

    topfile = glob.glob("run_*/c25_min_cent.pdb")[0]

    ini_xyz = []
    for i in range(len(samp_traj_idxs)):
        tic_traj_idx, frame_idx = samp_traj_idxs[i]
        run_idx, traj_idx = traj_idxs[tic_traj_idx]

        tname = "run_{}/c25_traj_cent_{}.dcd".format(run_idx, traj_idx)
        frame = md.load(tname, top=topfile)[tic_traj_idx]
        ini_xyz.append(frame.xyz)

    return ini_xyz 


def save_coords(saveas, xyz_ply):
    # create polymer topology
    topology = app.Topology()
    chain = topology.addChain()

    for i in range(len(xyz_ply)):
        res = topology.addResidue("PLY", chain)
        atm = topology.addAtom("PL", app.element.get_by_symbol("Pl"), res)
        if i == 0:
            prev_atm = atm
        else:
            topology.addBond(prev_atm, atm)
            prev_atm = atm

    positions = unit.Quantity(xyz_ply, unit.nanometer)

    topology.setUnitCellDimensions(box_edge.value_in_unit(unit.nanometer)*omm.Vec3(1, 1, 1))

    pdb = app.PDBFile("dum.pdb")
    with open(saveas, "w") as fout:
        pdb.writeFile(topology, positions, file=fout)


if __name__ == "__main__":
    msm_savedir = "../msm_dists"

    n_runs = 10
    run_idxs = np.arange(1, n_runs + 1)

    mass_ply = 37*unit.amu
    if "PL" not in app.element.Element._elements_by_symbol:
        app.element.polymer = app.element.Element(200, "Polymer", "Pl", mass_ply)

    print("sampling starting configurations...")
    cwd = os.getcwd()
    os.chdir(msm_savedir)
    ini_xyz = sample_starting_configurations(n_runs)
    os.chdir(cwd)

    print("saving...")
    for i in range(len(run_idxs)):
        rundir = "run_" + str(run_idxs[i])
        if not os.path.exists(rundir):
            os.mkdir(rundir)

        os.chdir(rundir)
        with open("dum.pdb", "w") as fout:
            fout.write(" ")
        save_coords("c25_noslv_min.pdb", ini_xyz[i])
        os.chdir("..")

    
