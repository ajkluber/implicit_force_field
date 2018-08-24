import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('name', type=str, help='Name.')
    parser.add_argument('subdir', type=str, help='Name.')
    args = parser.parse_args()

    name = args.name
    subdir = args.subdir

    import matplotlib as mpl
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'

    #name = "c25"
    savedir = "rg_dist"

    cwd = os.getcwd()

    if len(glob.glob(subdir + "/eps_*/T_*/" + savedir)) > 0:
        # doing several parameter sets
        epsdirs = glob.glob(subdir + "/eps_*")
    elif len(glob.glob(subdir + "/T_*/" + savedir)) > 0:
        # doing one parameter set
        epsdirs = [subdir]
    else:
        raise IOError("I don't see Rg data in this subdir")


    for i in range(len(epsdirs)):
        os.chdir(epsdirs[i])
        Tdirs = glob.glob("T_*/" + savedir + "/Pn.npy")
        T = [ float((x.split("/")[0])[2:]) for x in Tdirs ]
        T.sort()

        print os.getcwd() 
 
        plt.figure()
        avgRg = []
        for j in range(len(T)):
            # plot rg dist for T 
            avgRg.append(float(np.loadtxt("T_{:.2f}/{}/avg_Rg.dat".format(T[j],savedir))))
            mid_bin = np.load("T_{:.2f}/{}/mid_bin.npy".format(T[j],savedir))
            Pn = np.load("T_{:.2f}/{}/Pn.npy".format(T[j],savedir))
            if os.path.exists("T_{:.2f}/{}/dPn.npy".format(T[j],savedir)):
                dPn = np.load("T_{:.2f}/{}/dPn.npy".format(T[j],savedir))
            else:
                dPn = None

            if dPn is None:
                plt.plot(mid_bin, Pn, label=r"$T={:.2f}$".format(T[j]))
            else:
                plt.errorbar(mid_bin, Pn, yerr=dPn, label=r"$T={:.2f}$".format(T[j]))

            plt.legend(loc=1)

        if not os.path.exists("plots"):
            os.mkdir("plots")
        os.chdir("plots")
        plt.legend(loc=1)
        plt.xlabel(r"Radius of gyration (nm)")
        plt.ylabel(r"Prob. density")
        plt.title(epsdirs[i])
        plt.savefig("rg_dist_vs_T.pdf")
        plt.savefig("rg_dist_vs_T.png")

        plt.figure()
        plt.plot(T, avgRg)
        plt.xlabel("Temperature (K)")
        plt.ylabel(r"$\langle R_g \rangle$")
        plt.title(epsdirs[i])
        plt.savefig("rg_avg_vs_T.pdf")
        plt.savefig("rg_avg_vs_T.png")
        os.chdir(cwd)

    for i in range(len(epsdirs)):
        os.chdir(epsdirs[i])
        Tdirs = glob.glob("T_*/" + savedir + "/Pn.npy")
        T = [ float((x.split("/")[0])[2:]) for x in Tdirs ]
        T.sort()

        plt.figure()
        for j in range(len(T)):
            # plot rg dist for T 
            mid_bin = np.load("T_{:.2f}/{}/mid_bin.npy".format(T[j],savedir))
            Pn = np.load("T_{:.2f}/{}/Pn.npy".format(T[j],savedir))
            pmf = np.zeros(len(Pn), float)
            pmf[Pn > 0] = -np.log(Pn[Pn > 0])
            pmf -= pmf.min()

            #if os.path.exists("T_{:.2f}/{}/dPn.npy".format(T[j],savedir)):
            #    dPn = np.load("T_{:.2f}/{}/dPn.npy".format(T[j],savedir))
            #else:
            #    dPn = None

            #if dPn is None:
            plt.plot(mid_bin[Pn > 0], pmf[Pn > 0], label=r"$T={:.2f}$".format(T[j]))
            #else:
            #    plt.errorbar(mid_bin, Pn, yerr=dPn, label=r"$T={:.2f}$".format(T[j]))
            plt.ylim(0,4)

            plt.legend(loc=1)

        if not os.path.exists("plots"):
            os.mkdir("plots")
        os.chdir("plots")
        plt.legend(loc=1)
        plt.xlabel(r"Radius of gyration (nm)")
        plt.ylabel(r"Free energy $-\log P(R_g)$")
        plt.title(epsdirs[i])
        plt.savefig("rg_pmf_vs_T.pdf")
        plt.savefig("rg_pmf_vs_T.png")
        os.chdir(cwd)


    raise SystemExit

    plt.figure()
    # plot rg vs T for each value of eps_slv
    for i in range(len(eps_vals)):
        eps_ply, eps_slv = eps_vals[i]
        os.chdir("eps_ply_{:.2f}_eps_slv_{:.2f}".format(eps_ply, eps_slv))

        print os.getcwd()

        Tpaths = glob.glob("T_*")
        T = [ float((os.path.basename(x)).split("_")[1]) for x in Tpaths ]
        T.sort()
        hasT = []
        avgRg = []
        for j in range(len(T)):
            rg_file = "T_{:.2f}/{}/avg_Rg.dat".format(T[j], savedir)
            if os.path.exists(rg_file):
                avgRg.append(float(np.loadtxt(rg_file)))
                hasT.append(T[j])
        #avgRg = np.array([ float(np.loadtxt("T_{:.2f}/{}/avg_Rg.dat".format(T[j], savedir))) for j in range(len(T)) ])

        plt.plot(hasT, avgRg, 'o-', label=r"$\epsilon_{{ply}} = {:.2f}$ $\epsilon_{{slv}} = {:.2f}$".format(eps_ply, eps_slv))
        os.chdir("..")

    if not os.path.exists("plots"):
        os.mkdir("plots")

    os.chdir("plots")

    plt.legend(loc=1)
    plt.xlabel("Temperature")
    plt.ylabel("Rg")
    plt.title(subdir)
    plt.savefig("rg_vs_T.pdf")
    plt.savefig("rg_vs_T.png")

    os.chdir(cwd)
