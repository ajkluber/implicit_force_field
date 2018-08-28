import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('subdir', type=str, help='Subdirectory.')
    parser.add_argument('ti_idx', type=int, help='TICA index to plot.')
    parser.add_argument('eps_slv', type=float, nargs="+", help='Epsilon parameters to plot.')
    args = parser.parse_args()

    name = args.name
    subdir = args.subdir
    eps_slv_vals = args.eps_slv
    ti_idx = args.ti_idx

    #python ~/code/implicit_force_field/polymer_scripts/plot_tica_ti_vs_eps.py c25 c25_LJ_LJslv 0 0.1 0.5 1 2 3 4

    cwd = os.getcwd()

    os.chdir(subdir)
 
    if not os.path.exists("plots"):
        os.mkdir("plots")

    epsdirs = [ "eps_ply_0.10_eps_slv_{:.2f}".format(x) for x in eps_slv_vals ]

    savedir = "msm_dih_dists_rg"

    #if len(glob.glob(subdir + "/eps_*/T_*/" + savedir)) > 0:
    #    # doing several parameter sets
    #    epsdirs = glob.glob(subdir + "/eps_*")
    #elif len(glob.glob(subdir + "/T_*/" + savedir)) > 0:
    #    # doing one parameter set
    #    epsdirs = [subdir]
    #else:
    #    raise IOError("I don't see Rg data in this subdir")
    
    #ti_idx = 0
    #ti_idx = 1
    #ti_idx = 2

    plt.figure()
    for i in range(len(epsdirs)):
        os.chdir(epsdirs[i])
        Texist = [ float(x.split("/")[0].split("_")[1]) for x in glob.glob("T_*/{}/tica_ti.npy".format(savedir)) ]
        Texist.sort()

        ti = np.array([ np.load("T_{:.2f}/{}/tica_ti.npy".format(T, savedir))[ti_idx] for T in Texist])
        plt.plot(Texist, ti, 'o-', label=r"$\epsilon_{{ss}} = {:.2f}$".format(eps_slv_vals[i]))
        #avgRg = np.array([ float(np.loadtxt("eps_ply_{:.2f}/T_{:.2f}/{}/avg_Rg.dat".format(x, T, savedir))) for x in eps_ply_vals ])
        os.chdir("..")

    os.chdir("plots")
    plt.legend(loc=1)
    #plt.xlabel(r"$\epsilon_{ply}$ (kJ/mol)")
    plt.xlabel(r"Temperature (K)")
    plt.ylabel(r"TICA  $t_" + str(ti_idx+1) + "$ (frames)")
    #plt.title(subdir)
    plt.savefig("tica_t{}_vs_T_all_eps.pdf".format(ti_idx+1))
    plt.savefig("tica_t{}_vs_T_all_eps.png".format(ti_idx+1))
    os.chdir("..")

    raise SystemExit

    xmin, xmax = 0.3, 0.8

    fig, axes = plt.subplots(len(epsdirs), 1, sharex=True, figsize=(8, 3.5*len(epsdirs)))
    for i in range(len(epsdirs)):
        os.chdir(epsdirs[i])

        ax = axes[i]

        Texist = [ float(x.split("/")[0].split("_")[1]) for x in glob.glob("T_*/{}/avg_Rg.dat".format(savedir)) ]
        Texist.sort()

        for j in range(len(Texist)):
            mid_bin = np.load("T_{:.2f}/{}/mid_bin.npy".format(Texist[j], savedir))
            Pn = np.load("T_{:.2f}/{}/Pn.npy".format(Texist[j], savedir))
            if os.path.exists("T_{:.2f}/{}/dPn.npy".format(Texist[j], savedir)):
                dPn = np.load("T_{:.2f}/{}/dPn.npy".format(Texist[j], savedir))
                ax.errorbar(mid_bin, Pn, yerr=dPn, label=r"T = {:.2f}".format(Texist[j]))
            else:
                ax.plot(mid_bin, Pn, label=r"T = {:.2f}".format(Texist[j]))
        #ax.set_title(r"$\epsilon_{{ss}} = {:.2f}$".format(eps_slv_vals[i]))
        ax.annotate(r"$\epsilon_{{ss}} = {:.2f}$".format(eps_slv_vals[i]),
                fontsize=24,
                xy=(0,0), xytext=(0.5, 0.82), xycoords="axes fraction",
                textcoords="axes fraction")
        os.chdir("..")

        ax.set_xlim(xmin, xmax)
        if i == 2:
            ax.set_ylabel(r"$P(R_g)$")

        if i == (len(epsdirs) - 1):
            ax.set_xlabel(r"$R_g$ (nm)")
        ax.legend(loc=1)

    os.chdir("plots")
    fig.savefig("rg_dist_vs_T_all_eps.pdf")
    fig.savefig("rg_dist_vs_T_all_eps.png")
    os.chdir("..")

    #plt.figure()

    #cwd = os.getcwd()
    #for i in range(len(eps_ply_vals)):
    #    eps = eps_ply_vals[i]
    #    #os.chdir("eps_ply_{}_eps_slv_{:.2f}/T_{:.2f}/{}".format(ply_key, eps, T, savedir))
    #    os.chdir("eps_ply_{:.2f}/T_{:.2f}/{}".format(eps, T, savedir))

    #    if os.path.exists("Pn.npy"):
    #        Pn = np.load("Pn.npy")
    #        mid_bin = np.load("mid_bin.npy")

    #        if os.path.exists("dPn.npy"):
    #            dPn = np.load("dPn.npy")
    #            if eps == 0.10:
    #                plt.errorbar(mid_bin, Pn, fmt='o-', yerr=dPn, color="k", ecolor="k", label=r"$\epsilon_{{ply}} = {:.2f}$".format(eps))
    #            else:
    #                plt.errorbar(mid_bin, Pn, fmt='o-', label=r"$\epsilon_{{ply}} = {:.2f}$".format(eps))
    #        else:
    #            if eps == 0.10:
    #                plt.plot(mid_bin, Pn, 'ko-', label=r"$\epsilon_{{ply}} = {:.2f}$".format(eps))
    #            else:
    #                plt.plot(mid_bin, Pn, 'o-', label=r"$\epsilon_{{ply}} = {:.2f}$".format(eps))
    #    os.chdir(cwd)

    #os.chdir("plots")
    #ymin, ymax = plt.ylim()
    #plt.ylim(0, ymax)
    #plt.legend(loc=1)
    #plt.xlabel(r"$R_g$ (nm)")
    #plt.ylabel(r"Prob density $p(R_g)$")
    #plt.title(subdir)
    #plt.savefig("rg_dist_vs_eps.pdf")
    #plt.savefig("rg_dist_vs_eps.png")
    #os.chdir("..")
