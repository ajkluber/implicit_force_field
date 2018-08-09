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

    os.chdir(subdir)
    #Tpaths = glob.glob(subdir + "/*/T_*")
    Tpaths = glob.glob("T_*")

    cwd = os.getcwd()
    for i in range(len(Tpaths)):
        os.chdir(Tpaths[i])

        files = ["n.npy", "Pn.npy", "dPn.npy", "bin_edges.npy", "mid_bin.npy"]
        all_files_exist = np.all([ os.path.exists(savedir + "/" + x) for x in files ])

        no_data = False
        if all_files_exist:
            os.chdir(savedir)
            n = np.load("n.npy")
            Pn = np.load("Pn.npy")
            dPn = np.load("dPn.npy")
            bin_edges = np.load("bin_edges.npy")
            mid_bin = np.load("mid_bin.npy")
            os.chdir("..")
        else:
            rg_names = glob.glob("run_*/rg_*.npy")

            if len(rg_names) > 0:
                all_rg = []
                for x in rg_names:
                    # skip part of the first trajectory for each run as it might start
                    # far away from stationary distribution.
                    if os.path.basename(x) == "rg_1.npy":
                        all_rg.append(np.load(x)[200:])
                    else:
                        all_rg.append(np.load(x))
                max_rg = np.max([ np.max(x) for x in all_rg ])
                min_rg = np.min([ np.min(x) for x in all_rg ])

                bin_edges = np.linspace(min_rg, max_rg, 100)
                mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])
                n, _ = np.histogram(np.concatenate(all_rg), bins=bin_edges)
                Pn, _ = np.histogram(np.concatenate(all_rg), density=True, bins=bin_edges)

                dPn = np.std([ np.histogram(x, density=True, bins=bin_edges)[0] for x in all_rg ], axis=0)

                #import pdb
                #pdb.set_trace()

                if not os.path.exists(savedir):
                    os.mkdir(savedir)
                os.chdir(savedir)
                np.save("n.npy", n)
                np.save("Pn.npy", Pn)
                np.save("Pn.npy", Pn)
                np.save("bin_edges.npy", bin_edges)
                np.save("mid_bin.npy", mid_bin)
                os.chdir("..")
            else:
                print "No rg.npy for this temperature: ", Tpaths[i]
                no_data = True

        if not no_data:
            os.chdir(savedir)
            plt.figure()
            plt.errorbar(mid_bin, Pn, yerr=dPn, lw=2)
            plt.xlabel("Radius gyration (nm)")
            plt.ylabel("Prob density")
            plt.title(Tpaths[i])
            plt.savefig("Rg_dist.pdf")
            plt.savefig("Rg_dist.png")
            os.chdir("..")

        os.chdir(cwd)

