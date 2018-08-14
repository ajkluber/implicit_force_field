import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('logfile', type=str, help='Name.')
    args = parser.parse_args()

    logfile = args.logfile

    V = np.loadtxt(logfile, delimiter=",", usecols=(4,))
    E = np.loadtxt(logfile, delimiter=",", usecols=(1,))

    #target_volume = (5.9626)**3 = 211.986

    fig, ax1 = plt.subplots()
    ax1.plot([0,1], E[:2], color='g', label="Volume")
    ax1.plot(E, color='k', label="Energy")
    ax1.set_ylabel("Energy (kJ/mol)")
    ax1.legend(loc=1)

    ax2 = ax1.twinx()
    ax2.plot(V, color='g')
    ax2.set_ylabel(r"Volume (nm$^3$)")
    #ax2.axhline(target_vol, ls="--", color="grey") 
    ax1.set_xlabel("Step")

    fig.savefig("V_vs_step.pdf")
    fig.savefig("V_vs_step.png")

