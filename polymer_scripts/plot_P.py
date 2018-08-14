import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('target_vol', type=float, help='Target volume in nm^3.')
    args = parser.parse_args()

    target_vol = args.target_vol

    P = np.load("pressure_in_atm_vs_step.npy")
    V = np.load("volume_in_nm3_vs_step.npy")

    #target_volume = (5.9626)**3 = 211.986

    fig, ax1 = plt.subplots()
    ax1.plot([0,1], P[:2], color='g', label="Volume")
    ax1.plot(P, color='k', label="Pressure")
    ax1.set_ylabel("Pressure (atm)")
    ax1.legend(loc=7)

    ax2 = ax1.twinx()
    ax2.plot(V, color='g')
    ax2.set_ylabel(r"Volume (nm$^3$)")
    ax2.axhline(target_vol, ls="--", color="grey") 
    ax1.set_xlabel("Step")

    fig.savefig("P_vs_step.pdf")
    fig.savefig("P_vs_step.png")

