import os
import argparse
import glob

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['errorbar.capsize'] = 10
import matplotlib.pyplot as plt

def get_coeffs(gamma, s_list, bonds, angles, non_bond_wca, non_bond_gaussians, indv=False):

    # get bonded and wca coefficients
    c_vs_s = []
    if bonds or angles or non_bond_wca:
        for n in range(start_k):
            if indv:
                temp_c = []
                for i in range(len(s_list)):
                    coeff = gamma*np.load("coeff_{}_s_{}.npy".format(n + 1, s_list[i]))
                    temp_c.append(coeff)
            else:
                temp_c = gamma*np.load("coeff_{}_vs_s.npy".format(n + 1))
            c_vs_s.append(temp_c)

    # get gaussian coefficients
    c_vs_s_gauss = []
    if non_bond_gaussians: 
        for n in range(10):
            temp_c = []
            for i in range(len(s_list)):
                coeff = gamma*np.load("coeff_{}_s_{}.npy".format(start_k + n + 1, s_list[i]))
                temp_c.append(coeff)
            c_vs_s_gauss.append(temp_c)
    return c_vs_s, c_vs_s_gauss

def plot_bond_coeffs_vs_s(start_k, dt_frame, s_list, c_vs_s, coeff_true, line_colors, ylabels, ylog):

    if start_k > 1:
        fig, axes = plt.subplots(start_k, 1, figsize=(5, start_k*4))
        for n in range(start_k):
            ax = axes[n]
            avg_c = [ np.mean(c_vs_s[n][i]) for i in range(len(s_list)) ]
            err_c = [ np.std(c_vs_s[n][i])/np.sqrt(float(len(c_vs_s[n][i]))) for i in range(len(s_list)) ]
            ax.errorbar(dt_frame*s_list, avg_c, yerr=err_c, lw=3, color=line_colors[n], capsize=10, elinewidth=2, capthick=3)
            if ylog:
                ax.semilogy()
            ax.semilogx()
            ax.set_ylabel(ylabels[n], fontsize=20)
            ax.set_xlim(0.9*np.min(dt_frame*s_list), 1.1*np.max(dt_frame*s_list))
        ax.set_xlabel(r"$\Delta t$ (ps)", fontsize=20)
        plt.savefig("bond_coeffs_vs_s.pdf")
        plt.savefig("bond_coeffs_vs_s.png")
    elif start_k == 1:
        plt.figure()
        avg_c = [ np.mean(c_vs_s[0][i]) for i in range(len(s_list)) ]
        err_c = [ np.std(c_vs_s[0][i])/np.sqrt(float(len(c_vs_s[0][i]))) for i in range(len(s_list)) ]
        plt.errorbar(dt_frame*s_list, avg_c, yerr=err_c, lw=2, color=line_colors[2], capsize=10, elinewidth=2, capthick=3)
        plt.xlim(0.9*np.min(dt_frame*s_list), 1.1*np.max(dt_frame*s_list))
        if ylog:
            plt.semilogy()
        plt.semilogx()
        plt.xlabel(r"$\Delta t$ (ps)", fontsize=20)
        plt.ylabel(ylabels[2], fontsize=20)
        plt.savefig("bond_coeffs_vs_s.pdf")
        plt.savefig("bond_coeffs_vs_s.png")

    if len(coeff_true) > 0:
        plt.figure()
        for n in range(start_k):
            avg_c = [ np.mean(c_vs_s[n][i])/coeff_true[n] for i in range(len(s_list)) ]
            err_c = [ np.std(c_vs_s[n][i])/(coeff_true[n]*np.sqrt(float(len(c_vs_s[n][i])))) for i in range(len(s_list)) ]
            if len(coeff_true) == 1:
                plt.errorbar(dt_frame*s_list, avg_c, yerr=err_c, lw=3, color=line_colors[2], capsize=10, elinewidth=2, capthick=3, label=ylabels[2])
            else:
                plt.errorbar(dt_frame*s_list, avg_c, yerr=err_c, lw=3, color=line_colors[n], capsize=10, elinewidth=2, capthick=3, label=ylabels[n])
        legend = plt.legend(loc=1, fontsize=16, frameon=True, fancybox=False, framealpha=1)
        legend.get_frame().set_edgecolor("k")
        plt.ylabel("Effective / True", fontsize=20)
        plt.xlim(0.9*np.min(dt_frame*s_list), 1.1*np.max(dt_frame*s_list))
        if ylog:
            plt.semilogy()
        plt.semilogx()
        plt.xlabel(r"$\Delta t$ (ps)", fontsize=20)
        plt.savefig("bond_coeffs_over_true_vs_s.pdf")
        plt.savefig("bond_coeffs_over_true_vs_s.png")

def plot_gauss_coeffs(dt_frame, s_list, c_vs_s_gauss):
    plt.figure()
    for n in range(len(c_vs_s_gauss)):
        avg_c = [ np.mean(c_vs_s_gauss[n][i]) for i in range(len(s_list)) ]
        err_c = [ np.std(c_vs_s_gauss[n][i])/np.sqrt(float(len(c_vs_s_gauss[n][i]))) for i in range(len(s_list)) ]
        plt.errorbar(dt_frame*s_list, avg_c, yerr=err_c, capsize=10, elinewidth=2, capthick=3)
    plt.xlim(0.9*np.min(dt_frame*s_list), 1.1*np.max(dt_frame*s_list))
    plt.semilogx()
    plt.xlabel(r"$\Delta t$ (ps)", fontsize=20)
    plt.ylabel(r"Coefficient", fontsize=20)
    plt.savefig("gauss_coeffs_vs_s.pdf")
    plt.savefig("gauss_coeffs_vs_s.png")

def plot_effective_pair(non_bond_wca, line_colors, s_list, dt_frame, c_vs_s_gauss, dirsuff):

    plot_idxs = range(0, len(s_list), 1)
    saveas = "eff_pair{}_s_all".format(dirsuff)
    plot_eff_pair(non_bond_wca, plot_idxs, saveas, line_colors, s_list, dt_frame, c_vs_s_gauss, dirsuff)

    plot_idxs = [ i for i in range(len(s_list)) if s_list[i] in [1, 5, 20]]
    suffix = "_".join([ str(x) for x in plot_idxs])
    saveas = "eff_pair{}_s_{}".format(dirsuff, suffix)
    plot_eff_pair(non_bond_wca, plot_idxs, saveas, line_colors, s_list, dt_frame, c_vs_s_gauss, dirsuff)

def plot_eff_pair(non_bond_wca, plot_idxs, saveas, line_colors, s_list, dt_frame, c_vs_s_gauss, dirsuff):

    rmin = 3
    rmax = 10 
    gauss_r0 = [ (rmin + i)/10. for i in range(rmax) ]
    gauss_w = 0.1
    gauss_func = lambda r, r0, w: -np.exp(-0.5*((r - r0)/w)**2)
    r = np.linspace(0, 3, 1000)

    plt.figure()
    for i in range(len(plot_idxs)):
        s_idx = plot_idxs[i]
        s = s_list[s_idx]*dt_frame

        y = np.zeros(len(r), float)

        avg_c_vs_s_gauss = [] 
        for n in range(len(c_vs_s_gauss)):
            c_k = np.mean(c_vs_s_gauss[n][s_idx])
            y += c_k*gauss_func(r, gauss_r0[n], gauss_w)
        y += y.max()
        if i == 0:
            ymin = y.min()

        if non_bond_wca:
            plt.plot(r, y + WCA(r, 0.373, np.mean(c_vs_s[start_k - 1][s_idx])), lw=3, color=line_colors[i], label=r"$\Delta t = {}$ ps".format(s))
        else:
            plt.plot(r, y, lw=3, color=line_colors[i], label=r"$\Delta t = {}$ ps".format(s))
    legend = plt.legend(loc=4, fontsize=16, frameon=True, fancybox=False, framealpha=1)
    legend.get_frame().set_edgecolor("k")
    plt.xlim(0.2, 2)
    plt.ylim(1.4*ymin, -ymin)
    plt.xlabel("Long-range distance (nm)", fontsize=20)
    plt.ylabel("Effective interaction", fontsize=20)
    plt.savefig(saveas + ".pdf")
    plt.savefig(saveas + ".png")

    plt.figure()
    for n in range(len(gauss_r0)):
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

def WCA(r, sigma, eps):
    val = np.zeros(len(r), float)
    r0 = sigma*(2**(1./6))
    val[r < r0] = 4.*((sigma/r[r < r0])**12 - (sigma/r[r < r0])**6) + 1.
    return eps*val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('gamma', type=float, help='Friction coefficient.')
    parser.add_argument('dt_frame', type=float, help='Output timestep.')
    parser.add_argument('--coeff_true', nargs="+", default=[], type=float, help='True coefficients.')
    parser.add_argument('--ylog', action="store_true")
    parser.add_argument('--bonds', action="store_true")
    parser.add_argument('--angles', action="store_true")
    parser.add_argument('--non_bond_gaussians', action="store_true")
    parser.add_argument('--non_bond_wca', action="store_true")

    n_beads = 25

    args = parser.parse_args()
    gamma = args.gamma
    dt_frame = args.dt_frame
    coeff_true = args.coeff_true
    ylog = args.ylog
    bonds = args.bonds
    angles = args.angles
    non_bond_wca = args.non_bond_wca
    non_bond_gaussians = args.non_bond_gaussians

    ylabels = [r"$k_b$", r"$k_a$", r"$\epsilon$"]
    line_colors = ["#ff7f00", "#377eb8", "grey", "#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#e41a1c", "#4daf4a", "#984ea3"] # orange, blue, grey

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
    if len(s_list) > 0:
        s_list.sort()
        s_list = np.array(s_list)
        indv = True
    else:
        s_list = np.load("s_list.npy")
        indv = False

    c_vs_s, c_vs_s_gauss = get_coeffs(gamma, s_list, bonds, angles, non_bond_wca, non_bond_gaussians, indv=indv)

    # plot coefficients versus s
    if bonds or angles or non_bond_wca:
        plot_bond_coeffs_vs_s(start_k, dt_frame, s_list, c_vs_s, coeff_true, line_colors, ylabels, ylog)

    if non_bond_gaussians:
        plot_gauss_coeffs(dt_frame, s_list, c_vs_s_gauss)

    if non_bond_gaussians:
        plot_effective_pair(non_bond_wca, line_colors, s_list, dt_frame, c_vs_s_gauss, dirsuff)

    os.chdir("..")
