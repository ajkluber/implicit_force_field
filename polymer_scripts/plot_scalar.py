import os
import numpy as np 

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

def plot_scalar():
    psi_fj = np.load("psi_fj.npy")[0]
    psi_gU0_fj = np.load("psi_gU0_fj.npy")[0]
    psi_gU1_fj = np.load("psi_gU1_fj.npy")[0]
    psi_Lap_fj = np.load("psi_Lap_fj.npy")[0]
    psi_Gen_fj = np.load("psi_Gen_fj.npy")[0]

    x = np.arange(len(psi_gU0_fj))

    A = np.array([psi_gU0_fj, psi_gU1_fj, psi_Lap_fj]).T
    coeff = np.linalg.lstsq(A, psi_fj)[0]

    temp_Gen = np.einsum("k,ik->i", coeff, A)

    kappa = 1./np.load("../msm_dists/tica_ti_ps.npy")[0]

    import scipy.optimize
    xdata = -kappa*psi_fj
    ydata = psi_Gen_fj
    popt, pcov = scipy.optimize.curve_fit(lambda x, m, b: m*x + b, xdata, ydata, p0=(1,0))

    #plt.figure()
    #plt.plot(x, temp_Gen)
    #plt.plot(x, psi_fj)
    #plt.xlabel(r"$f_j$ center")
    #plt.savefig("scalar_fit.pdf")
    #plt.savefig("scalar_fit.png")
    
    #plt.figure()
    #plt.plot(psi_gU0_fj - psi_gU1_fj + psi_Lap_fj)
    #plt.xlabel(r"$f_j$ center")
    #plt.savefig("scalar_2.pdf")
    #plt.savefig("scalar_2.png")

    scale = 1000 

    x_fit = np.linspace(xdata.min(), xdata.max(), 100)
    y_fit = popt[0]*x_fit + popt[1]

    crr = np.corrcoef(xdata, ydata)[0,1]
    plt.figure()
    plt.plot(scale*xdata, scale*ydata, 'o') 
    plt.annotate(r"$\rho = {:.3f}$".format(crr), xy=(0,0), xytext=(0.1, 0.5),
            xycoords="axes fraction", textcoords="axes fraction", fontsize=18)
    plt.plot(scale*x_fit, scale*y_fit, color="k",
            ls="--", label=r"$y = {:.3f} x + {:.3f}$".format(popt[0], scale*popt[1]))
    plt.legend()
            
    plt.xlabel(r"$-\kappa_1\langle \psi_1 | f_j \rangle$ x" + str(scale))
    plt.ylabel(r"$\langle \psi_1 | \mathcal{L} f_j \rangle$ x" + str(scale))
    plt.savefig("scatter_psi_Gen_fj_vs_psi_fj.pdf")
    plt.savefig("scatter_psi_Gen_fj_vs_psi_fj.png")

    plt.figure()
    plt.plot(x, -scale*kappa*psi_fj, 'ko', label=r"$-\kappa_1\langle \psi_1 | f_j \rangle$") 
    plt.plot(x, scale*psi_Gen_fj, 'ro', label=r"$\langle \psi_1 | \mathcal{L} f_j \rangle$") 
    plt.xlabel(r"$f_j$ center")
    plt.ylabel(r"x" + str(scale))
    plt.legend()
    #plt.ylabel(r"$\langle \psi_1 | f_j \rangle$")
    plt.savefig("scalar_psi_Gen_fj.pdf")
    plt.savefig("scalar_psi_Gen_fj.png")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 13))
    ax1.plot(x, psi_gU0_fj, 'ro')
    ax2.plot(x, psi_gU1_fj, 'bo')
    ax3.plot(x, psi_Lap_fj, 'go')

    ax1.set_ylabel(r"$\langle \psi_1 | -\nabla U_0 \cdot \nabla f_j\rangle$")
    ax2.set_ylabel(r"$\langle \psi_1 | -\nabla U_1 \cdot \nabla f_j\rangle$")
    ax3.set_ylabel(r"$\langle \psi_1 | \Delta f_j\rangle$")
    ax3.set_xlabel(r"$f_j$ center")

    fig.savefig("scalar_gU0_gU1_Lap_fj.pdf")
    fig.savefig("scalar_gU0_gU1_Lap_fj.png")

if __name__ == "__main__":
    plot_scalar()
