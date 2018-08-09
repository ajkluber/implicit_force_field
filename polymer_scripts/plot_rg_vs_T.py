import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    cwd = os.getcwd()

    os.chdir(subdir)
 
    eps_paths = glob.glob("eps_slv_*")
    eps_vals = [ float(os.path.basename(x).split("_")[-1]) for x in eps_paths ]

    plt.figure()
    # plot rg vs T for each value of eps_slv
    for i in range(len(eps_vals)):
        os.chdir("eps_slv_{:.2f}".format(eps_vals[i]))

        Tpaths = glob.glob("T_*")
        T = [ float((os.path.basename(x)).split("_")[1]) for x in Tpaths ]
        T.sort()
        avgRg = np.array([ float(np.loadtxt("T_{:.2f}/{}/avg_Rg.dat".format(T[j], savedir))) for j in range(len(T)) ])

        plt.plot(T, avgRg, 'o-', label=r"$\epsilon_{{slv}} = {:.2f}$".format(eps_vals[i]))
        os.chdir("..")

    if not os.path.exists("plots"):
        os.mkdir("plots")

    os.chdir("plots")
    print os.getcwd()

    plt.legend(loc=2)
    plt.xlabel("Temperature")
    plt.ylabel("Rg")
    plt.title(subdir)
    plt.savefig("rg_vs_T.pdf")
    plt.savefig("rg_vs_T.png")

    os.chdir(cwd)
