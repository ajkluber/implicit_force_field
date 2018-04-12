import os
import numpy as np

import util 

if __name__ == "__main__":
    #trajfile = "c25_traj_ply_1.dcd"
    #topfile = "c25_min_ply_1.pdb"
    #dt_frame = 0.2
    #trajfile = "c25_traj_1.dcd"
    trajfile = "chunk_1.dcd"
    topfile = "c25_min_1.pdb"
    dt_frame = 0.0002

    kT = 0.0083145*300
    beta = 1/kT

    n_frames_tot, n_dim = util.get_n_frames(trajfile, topfile)

    savedir = "diffusion_vs_s"
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    all_s = [1, 2, 5, 10, 20, 50, 100, 500]
    for i in range(len(all_s)):
        print " ({}/{})".format(i + 1, len(all_s)) 
        s_frames = all_s[i]
        s = dt_frame*s_frames
        A = util.calc_diffusion(trajfile, topfile, beta, s_frames, s, n_dim, n_frames_tot)
        np.save("{}/A_{}.npy".format(savedir, s_frames), A)

    all_A = [ np.load("{}/A_{}.npy".format(savedir, all_s[i])) for i in range(len(all_s)) ]
