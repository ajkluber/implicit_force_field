from __future__ import print_function
import os
import sys
import glob
import argparse
import numpy as np 

from scipy.stats import binned_statistic_2d as bin2d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

if __name__ == "__main__":

    psi_fj = np.load("psi_fj.npy")    
    psi_gU0_fj = np.load("psi_gU0_fj.npy")
    psi_gU1_fj = np.load("psi_gU1_fj.npy")
    psi_Lap_fj = np.load("psi_Lap_fj.npy")
    psi_Gen_fj = np.load("psi_Gen_fj.npy")

    x = np.arange(len(psi_gU0_fj[0]))

    crr = np.corrcoef(psi_Gen_fj[0], psi_fj[0])[0,1]
    plt.figure()
    plt.plot(psi_Gen_fj[0], psi_fj[0], 'ko') 
    plt.annotate("corr = {:.3f}".format(crr), xy=(0,0), xytext=(0.2, 0.2),
            xycoords="axes fraction", textcoords="axes fraction")
    plt.xlabel(r"$\langle \psi_1 | \mathcal{L} f_j \rangle$")
    plt.ylabel(r"$\langle \psi_1 | f_j \rangle$")
    plt.savefig("scatter_psi_Gen_fj_vs_psi_fj.pdf")
    plt.savefig("scatter_psi_Gen_fj_vs_psi_fj.png")

    plt.figure()
    plt.plot(x, psi_fj[0], 'ko', label=r"$\langle \psi_1 | f_j \rangle$") 
    plt.plot(x, psi_Gen_fj[0], 'ro', label=r"$\langle \psi_1 | \mathcal{L} f_j \rangle$") 
    plt.xlabel(r"$f_j$ center")
    plt.legend()
    #plt.ylabel(r"$\langle \psi_1 | f_j \rangle$")
    plt.savefig("scalar_psi_Gen_fj.pdf")
    plt.savefig("scalar_psi_Gen_fj.png")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 13))
    ax1.plot(x, psi_gU0_fj[0], 'ro')
    ax2.plot(x, psi_gU1_fj[0], 'bo')
    ax3.plot(x, psi_Lap_fj[0], 'go')

    ax1.set_ylabel(r"$\langle \psi_1 | -\nabla U_0 \cdot \nabla f_j\rangle$")
    ax2.set_ylabel(r"$\langle \psi_1 | -\nabla U_1 \cdot \nabla f_j\rangle$")
    ax3.set_ylabel(r"$\langle \psi_1 | \Delta f_j\rangle$")
    ax3.set_xlabel(r"$f_j$ center")

    fig.savefig("scalar_gU0_gU1_Lap_fj.pdf")
    fig.savefig("scalar_gU0_gU1_Lap_fj.png")
