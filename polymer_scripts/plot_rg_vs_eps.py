import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def dummy():
    #eps_paths = glob.glob("eps_slv_*")
    #eps_paths = glob.glob("eps_ply_0.10_eps_slv_*")
    #eps_vals = [ float(os.path.basename(x).split("_")[-1]) for x in eps_paths ]
    #eps_paths = glob.glob("eps_ply_*_eps_slv_*")
    eps_paths = glob.glob("eps_ply_*")
    #eps_paths.pop(eps_paths.index("eps_ply_0.10_eps_slv_4.00"))

    eps_vals = {}
    for i in range(len(eps_paths)):
        eps_ply = float(os.path.basename(eps_paths[i]).split("_")[2])
        eps_slv = float(os.path.basename(eps_paths[i]).split("_")[-1])

        ply_key = "{:.2f}".format(eps_ply)
        if eps_vals.has_key(ply_key):
            eps_vals[ply_key].append(eps_slv)
        else:
            eps_vals[ply_key] = [eps_slv]

    if not os.path.exists("plots"):
        os.mkdir("plots")

    plt.figure()
    for ply_key in eps_vals.keys():
        eps_slv_vals = eps_vals[ply_key]
        eps_slv_vals.sort()
        #eps_vals[key] = temp

        #avgRg = np.array([ float(np.loadtxt("eps_ply_{}_eps_slv_{:.2f}/T_{:.2f}/{}/avg_Rg.dat".format(ply_key, x, T, savedir))) for x in eps_slv_vals ])
        avgRg = np.array([ float(np.loadtxt("eps_ply_{}/T_{:.2f}/{}/avg_Rg.dat".format(ply_key, x, T, savedir))) for x in eps_slv_vals ])
        plt.plot(eps_slv_vals, avgRg, 'o-')

    os.chdir("plots")
    plt.legend(loc=1)
    plt.xlabel(r"Solvent Interaction $\epsilon_{slv}$ (kJ/mol)")
    plt.ylabel(r"$\langle R_g \rangle$ (nm)")
    plt.title(subdir)
    plt.savefig("rg_vs_eps.pdf")
    plt.savefig("rg_vs_eps.png")
    os.chdir("..")

    raise SystemExit

    plt.figure()
    ply_key = "0.10"
    eps_slv_vals = eps_vals[ply_key]
    eps_slv_vals.sort()
    cwd = os.getcwd()
    for i in range(len(eps_slv_vals)):
        eps = eps_slv_vals[i]
        #os.chdir("eps_ply_{}_eps_slv_{:.2f}/T_{:.2f}/{}".format(ply_key, eps, T, savedir))
        os.chdir("eps_ply_{}/T_{:.2f}/{}".format(ply_key, eps, T, savedir))

        if os.path.exists("Pn.npy"):
            Pn = np.load("Pn.npy")
            mid_bin = np.load("mid_bin.npy")

            if os.path.exists("dPn.npy"):
                dPn = np.load("dPn.npy")
                if eps == 0.10:
                    plt.errorbar(mid_bin, Pn, fmt='o-', yerr=dPn, color="k", ecolor="k", label=r"$\epsilon_{{slv}} = {:.2f}$".format(eps))
                else:
                    plt.errorbar(mid_bin, Pn, fmt='o-', label=r"$\epsilon_{{slv}} = {:.2f}$".format(eps))
            else:
                if eps == 0.10:
                    plt.plot(mid_bin, Pn, 'ko-', label=r"$\epsilon_{{slv}} = {:.2f}$".format(eps))
                else:
                    plt.plot(mid_bin, Pn, 'o-', label=r"$\epsilon_{{slv}} = {:.2f}$".format(eps))
        os.chdir(cwd)

    os.chdir("plots")
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    plt.legend(loc=1)
    plt.xlabel(r"$R_g$ (nm)")
    plt.ylabel(r"Prob density $p(R_g)$")
    plt.title(subdir)
    plt.savefig("rg_dist_vs_eps.pdf")
    plt.savefig("rg_dist_vs_eps.png")
    os.chdir("..")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('subdir', type=str, help='Name.')
    args = parser.parse_args()

    subdir = args.subdir

    import matplotlib as mpl
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'

    name = "c25"
    savedir = "rg_dist"
    T = 300

    os.chdir(subdir)
 
    #eps_paths = glob.glob("eps_slv_*")
    #eps_paths = glob.glob("eps_ply_0.10_eps_slv_*")
    #eps_vals = [ float(os.path.basename(x).split("_")[-1]) for x in eps_paths ]
    #eps_paths = glob.glob("eps_ply_*_eps_slv_*")
    eps_paths = glob.glob("eps_ply_*")
    #eps_paths.pop(eps_paths.index("eps_ply_0.10_eps_slv_4.00"))

    eps_ply_vals = [ float(x.split("_")[2]) for x in eps_paths ]
    eps_ply_vals.sort()

    if not os.path.exists("plots"):
        os.mkdir("plots")

    plt.figure()

    avgRg = np.array([ float(np.loadtxt("eps_ply_{:.2f}/T_{:.2f}/{}/avg_Rg.dat".format(x, T, savedir))) for x in eps_ply_vals ])
    plt.plot(eps_ply_vals, avgRg, 'o-')

    os.chdir("plots")
    plt.legend(loc=1)
    plt.xlabel(r"$\epsilon_{ply}$ (kJ/mol)")
    plt.ylabel(r"$\langle R_g \rangle$ (nm)")
    plt.title(subdir)
    plt.savefig("rg_vs_eps.pdf")
    plt.savefig("rg_vs_eps.png")
    os.chdir("..")

    plt.figure()

    cwd = os.getcwd()
    for i in range(len(eps_ply_vals)):
        eps = eps_ply_vals[i]
        #os.chdir("eps_ply_{}_eps_slv_{:.2f}/T_{:.2f}/{}".format(ply_key, eps, T, savedir))
        os.chdir("eps_ply_{:.2f}/T_{:.2f}/{}".format(eps, T, savedir))

        if os.path.exists("Pn.npy"):
            Pn = np.load("Pn.npy")
            mid_bin = np.load("mid_bin.npy")

            if os.path.exists("dPn.npy"):
                dPn = np.load("dPn.npy")
                if eps == 0.10:
                    plt.errorbar(mid_bin, Pn, fmt='o-', yerr=dPn, color="k", ecolor="k", label=r"$\epsilon_{{ply}} = {:.2f}$".format(eps))
                else:
                    plt.errorbar(mid_bin, Pn, fmt='o-', label=r"$\epsilon_{{ply}} = {:.2f}$".format(eps))
            else:
                if eps == 0.10:
                    plt.plot(mid_bin, Pn, 'ko-', label=r"$\epsilon_{{ply}} = {:.2f}$".format(eps))
                else:
                    plt.plot(mid_bin, Pn, 'o-', label=r"$\epsilon_{{ply}} = {:.2f}$".format(eps))
        os.chdir(cwd)

    os.chdir("plots")
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    plt.legend(loc=1)
    plt.xlabel(r"$R_g$ (nm)")
    plt.ylabel(r"Prob density $p(R_g)$")
    plt.title(subdir)
    plt.savefig("rg_dist_vs_eps.pdf")
    plt.savefig("rg_dist_vs_eps.png")
    os.chdir("..")
