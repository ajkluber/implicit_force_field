import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import binned_statistic as bin1d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('subdir', type=str, help='Name.')
    args = parser.parse_args()

    subdir = args.subdir

    import matplotlib as mpl
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'

    name = "c25"
    #savedir = "rg_dist"
    savedir = "E_vs_rg"

    os.chdir(subdir)
    if len(glob.glob("*/T_*")) > 0:
        # We are in directory above temps.
        Tpaths = glob.glob("*/T_*")
    else:
        Tpaths = glob.glob("T_*")

    cwd = os.getcwd()
    for i in range(len(Tpaths)):
        os.chdir(Tpaths[i])

        files = ["avgE_vs_rg.npy", "bin_edges.npy", "mid_bin.npy", "bin_counts.npy"]
        all_files_exist = np.all([ os.path.exists(savedir + "/" + x) for x in files ])

        no_data = False
        if all_files_exist:
            os.chdir(savedir)
            avgE_vs_rg = np.load("avgE_vs_rg.npy") 
            bin_edges = np.load("bin_edges.npy") 
            mid_bin = np.load("mid_bin.npy")
            bin_counts = np.save("bin_counts.npy")
            os.chdir("..")
        else:
            rg_names = glob.glob("run_*/rg_*.npy")
            #E_names = glob.glob("run_*/{}_*.log".format(name))


            if len(rg_names) > 0:
                all_rg = []
                all_E = []
                for i in range(len(rg_names)):
                    # skip part of the first trajectory for each run as it might start
                    # far away from stationary distribution.
                    dirname = os.path.dirname(rg_names[i])
                    #import pdb 
                    #pdb.set_trace()

                    idx = rg_names[i][:-4].split("_")[2]
                    #idx = rg_names[i][:-4].split("_")[1]
                    E_name = dirname + "/" + name + "_" + idx + ".log"

                    if os.path.exists(rg_names[i]) and os.path.exists(E_name):
                        temp_rg = np.load(rg_names[i])
                        temp_E = np.loadtxt(E_name, delimiter=",", usecols=(1,))
                        #print len(temp_rg), len(temp_E)

                        #if os.path.basename(x) == "rg_1.npy":
                        if idx == "1":
                            all_rg.append(temp_rg[200:])
                            all_E.append(temp_E[200:])
                        else:
                            all_rg.append(temp_rg)
                            all_E.append(temp_E)
                max_rg = np.max([ np.max(x) for x in all_rg ])
                min_rg = np.min([ np.min(x) for x in all_rg ])

                bin_edges = np.linspace(min_rg, max_rg, 100)
                mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])

                avgE_vs_rg, _, _ = bin1d(np.concatenate(all_rg), np.concatenate(all_E), bins=bin_edges, statistic="mean")
                bin_counts, _, _ = bin1d(np.concatenate(all_rg), np.concatenate(all_E), bins=bin_edges, statistic="count")

                if not os.path.exists(savedir):
                    os.mkdir(savedir)
                os.chdir(savedir)
                np.save("avgE_vs_rg.npy", avgE_vs_rg)
                np.save("bin_edges.npy", bin_edges)
                np.save("mid_bin.npy", mid_bin)
                np.save("bin_counts.npy", bin_counts)
                os.chdir("..")
            else:
                print "No rg.npy for this temperature: ", Tpaths[i]
                no_data = True

        if not no_data:
            os.chdir(savedir)
            plt.figure()
            use = bin_counts > 20
            plt.plot(mid_bin[use], avgE_vs_rg[use], lw=2)
            plt.xlabel("Radius gyration (nm)")
            plt.ylabel(r"$\langle E \rangle$ (kJ/mol)")
            plt.title(Tpaths[i])
            plt.savefig("E_vs_rg.pdf")
            plt.savefig("E_vs_rg.png")
            os.chdir("..")

        os.chdir(cwd)

