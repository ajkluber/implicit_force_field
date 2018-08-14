import os
import argparse
import glob
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('subdir', type=str, help='Target volume in nm^3.')
    args = parser.parse_args()

    subdir = args.subdir

    os.chdir(subdir)
    cwd = os.getcwd()
    pdirs = glob.glob("*/pressure_equil")
    for i in range(len(pdirs)):
        os.chdir(pdirs[i])

        all_P = np.load("pressure_in_atm_vs_step.npy")
        all_V = np.load("volume_in_nm3_vs_step.npy")

        N = len(all_P)

        avgV = np.mean(all_V[N/2:]) 
        stdV = np.std(all_V[N/2:]) 
        avgP = np.mean(all_P[N/2:]) 
        stdP = np.std(all_P[N/2:]) 

        np.savetxt("avgV.dat", np.array([avgV, stdV]))
        np.savetxt("pressure.dat", np.array([avgP, stdP]))

        os.chdir(cwd)
