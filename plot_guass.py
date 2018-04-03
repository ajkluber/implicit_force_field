import os
import glob

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt


if __name__ == "__main__":
    gamma = 1
    #savedir = "coeff_vs_s_gauss"
    #savedir = "coeff_vs_s_bond_angle_wca_gauss"
    n_beads = 25

    bonds = True
    angles = True
    non_bond_wca = True
    non_bond_gaussians = True

    start_k = 0
    savedir = "coeff_vs_s"
    dirsuff = "" 
    if bonds:
        start_k += 1
        dirsuff += "_bond"
    if angles:
        start_k += 1
        dirsuff += "_angle"
    if non_bond_wca:
        start_k += 1
        dirsuff += "_wca"
    if non_bond_gaussians:
        dirsuff += "_gauss"
    savedir += dirsuff

    os.chdir(savedir)

    s_list = [ int(x.split("_")[3].split(".npy")[0]) for x in glob.glob("coeff_1_s_*npy") ]
    s_list.sort()
    s_list = np.array(s_list)

    c_vs_s = []
    #for n in range(start_k + 10):
    for n in range(start_k):
        temp_c = []
        for i in range(len(s_list)):
            coeff = np.load("coeff_{}_s_{}.npy".format(n + 1, s_list[i]))
            temp_c.append(coeff)
        c_vs_s.append(temp_c)

    c_vs_s_gauss = []
    for n in range(10):
        temp_c = []
        for i in range(len(s_list)):
            coeff = np.load("coeff_{}_s_{}.npy".format(start_k + n + 1, s_list[i]))
            temp_c.append(coeff)
        c_vs_s_gauss.append(temp_c)

    ylabels = [r"$k_b$", r"$k_a$", r"$\epsilon$"]

    if start_k > 1:
        fig, axes = plt.subplots(start_k, 1, figsize=(5, start_k*4))
        for n in range(start_k):
            ax = axes[n]
            avg_c = [ np.mean(c_vs_s[n][i]) for i in range(len(s_list)) ]
            err_c = [ np.std(c_vs_s[n][i])/np.sqrt(float(len(c_vs_s[n][i]))) for i in range(len(s_list)) ]
            ax.errorbar(0.2*s_list, avg_c, yerr=err_c)

            ax.set_ylabel(ylabels[n], fontsize=20)
        ax.set_xlabel(r"$\Delta t$ (ps)", fontsize=20)
        plt.savefig("bond_coeffs_vs_s.pdf")
        plt.savefig("bond_coeffs_vs_s.png")
    elif start_k == 1:
        fig, ax = plt.subplots(start_k, 1, figsize=(5, start_k*4))
        avg_c = [ np.mean(c_vs_s[0][i]) for i in range(len(s_list)) ]
        err_c = [ np.std(c_vs_s[0][i])/np.sqrt(float(len(c_vs_s[0][i]))) for i in range(len(s_list)) ]
        ax.errorbar(0.2*s_list, avg_c, yerr=err_c)

        ax.set_ylabel(ylabels[2], fontsize=20)
        ax.set_xlabel(r"$\Delta t$ (ps)", fontsize=20)
        plt.savefig("bond_coeffs_vs_s.pdf")
        plt.savefig("bond_coeffs_vs_s.png")

    plt.figure()
    for n in range(10):
        avg_c = [ np.mean(c_vs_s_gauss[n][i]) for i in range(len(s_list)) ]
        err_c = [ np.std(c_vs_s_gauss[n][i])/np.sqrt(float(len(c_vs_s_gauss[n][i]))) for i in range(len(s_list)) ]
        if n in [0, 1, 2]:
            plt.errorbar(0.2*s_list, avg_c, yerr=err_c, ls='--')
        else:
            plt.errorbar(0.2*s_list, avg_c, yerr=err_c)
    plt.xlabel(r"$\Delta t$ (ps)", fontsize=20)
    plt.ylabel(r"Coefficient $c_k$", fontsize=20)
    plt.savefig("coeffs_vs_s.pdf")
    plt.savefig("coeffs_vs_s.png")

    n_gauss = 10
    rmin = 3
    rmax = 10 
    gauss_r0 = [ (rmin + i)/10. for i in range(rmax) ]
    gauss_w = 1./10
    scale_factor = 1./10

    gauss_func = lambda r, r0, w: -np.exp(-0.5*((r - r0)/w)**2)

    def WCA(r, sigma, eps):
        val = np.zeros(len(r), float)
        r0 = sigma*(2**(1./6))
        val[r < r0] = 4.*((sigma/r[r < r0])**12 - (sigma/r[r < r0])**6) + 1.
        return eps*val

    r = np.linspace(0, 3, 1000)

    # plot basis functions
    if False:
        plt.figure()
        for n in range(10):
            #plt.plot(r, gauss_func(r, gauss_r0[n], gauss_w), 'k--', lw=3)
            plt.plot(r, gauss_func(r, gauss_r0[n], gauss_w), lw=3)
        plt.yticks([])
        plt.ylabel("Gaussian basis", fontsize=20)
        plt.xlabel("Long-range distance (nm)", fontsize=20)
        #plt.xlim(0.2, 2)
        plt.xlim(0, 1.5)
        plt.ylim(-1.1, 0.1)
        plt.savefig("gaussian_basis_colors.pdf")
        plt.savefig("gaussian_basis_colors.png")
        #plt.savefig("gaussian_basis.pdf")
        #plt.savefig("gaussian_basis.png")


    plot_all = True
    if plot_all:
        #plot_idxs = range(0, len(s_list), 2)
        plot_idxs = range(0, len(s_list), 1)
    else:
        plot_idxs = [ i for i in range(len(s_list)) if s_list[i] in [1, 5, 20]]

    line_colors = ["#ff7f00", "#377eb8", "grey", "#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#e41a1c", "#4daf4a", "#984ea3"] # orange, blue, grey

    plt.figure()
    for i in range(len(plot_idxs)):
        s_idx = plot_idxs[i]
        s = s_list[s_idx]*0.2

        y = np.zeros(len(r), float)

        avg_c_vs_s_gauss = [] 
        for n in range(10):
            c_k = np.mean(c_vs_s_gauss[n][s_idx])
            y += c_k*gauss_func(r, gauss_r0[n], gauss_w)
        y += y.max()
        if i == 0:
            ymin = y.min()

        if non_bond_wca:
            plt.plot(r, y + WCA(r, 0.373, np.mean(c_vs_s[start_k - 1][s_idx])), lw=3, color=line_colors[i], label=r"$\Delta t = {}$".format(s))
        else:
            plt.plot(r, y, lw=3, color=line_colors[i], label=r"$\Delta t = {}$".format(s))
    legend = plt.legend(loc=4, fontsize=16, frameon=True, fancybox=False, framealpha=1)
    legend.get_frame().set_edgecolor("k")
    plt.xlim(0.2, 2)
    #plt.ylim(-10, 0)
    #ymin, ymax = plt.ylim()
    plt.ylim(1.4*ymin, -ymin)
    plt.xlabel("Long-range distance (nm)", fontsize=20)
    plt.ylabel("Effective interaction", fontsize=20)
    if plot_all:
        saveas = "eff_pair{}_s_all".format(dirsuff)
    else:
        suffix = "_".join([ str(x) for x in plot_idxs])
        saveas = "eff_pair{}_s_{}".format(dirsuff, suffix)
    plt.savefig(saveas + ".pdf")
    plt.savefig(saveas + ".png")

    #plt.figure()
    ##for i in range(len(s_list)):
    #for i in [5,6,7]:
    #    s = s_list[i]*0.2

    #    y = np.zeros(len(r), float)

    #    avg_c_vs_s_gauss = [] 
    #    for n in range(10):
    #        c_k = np.mean(c_vs_s_gauss[n][i])
    #        y += scale_factor*c_k*gauss_func(r, gauss_r0[n], gauss_w)
    #    y += y.max()
    #    y /= 100.

    #    plt.plot(r, y, lw=3, label=r"$\Delta t = {}$".format(s))
    #legend = plt.legend(loc=4, fontsize=16, frameon=True, fancybox=False, framealpha=1)
    #legend.get_frame().set_edgecolor("k")
    ##plt.xlim(0.2, 1.8)
    #plt.xlim(0, 1.8)
    #plt.ylim(-0.8, 0)
    #plt.xlabel("Long-range distance (nm)", fontsize=20)
    #plt.ylabel("Effective interaction", fontsize=20)
    #plt.savefig("gauss_interaction_vs_long_s.pdf")
    #plt.savefig("gauss_interaction_vs_long_s.png")

    #plt.show()
