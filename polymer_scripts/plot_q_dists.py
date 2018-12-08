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
    savedir = "q_dist"

    os.chdir(subdir)
    if len(glob.glob("*/T_*")) > 0:
        # We are in directory above temps.
        Tpaths = glob.glob("*/T_*")
    else:
        Tpaths = glob.glob("T_*")

    cwd = os.getcwd()
    for i in range(len(Tpaths)):
        os.chdir(Tpaths[i])

        files = ["n.npy", "Pn.npy", "avg_Q.dat", "bin_edges.npy", "mid_bin.npy"]
        all_files_exist = np.all([ os.path.exists(savedir + "/" + x) for x in files ])

        no_data = False
        if all_files_exist:
            os.chdir(savedir)
            n = np.load("n.npy")
            Pn = np.load("Pn.npy")
            Pn = np.load("Pn.npy")
            avgQ = float(np.loadtxt("avg_Q.dat"))
            bin_edges = np.load("bin_edges.npy")
            mid_bin = np.load("mid_bin.npy")

            if os.path.exists("dPn.npy"):
                dPn = np.load("dPn.npy")
            else:
                dPn = None
            os.chdir("..")
        else:
            q_names = glob.glob("run_*/q_*.npy")

            if len(q_names) > 0:
                all_q = []
                for x in q_names:
                    # skip part of the first trajectory for each run as it might start
                    # far away from stationary distribution.
                    if os.path.basename(x) == "q_1.npy":
                        all_q.append(np.load(x)[200:])
                    else:
                        all_q.append(np.load(x))
                max_q = np.max([ np.max(x) for x in all_q ])
                min_q = np.min([ np.min(x) for x in all_q ])

                bin_edges = np.linspace(min_q, max_q, 100)
                mid_bin = 0.5*(bin_edges[1:] + bin_edges[:-1])
                n, _ = np.histogram(np.concatenate(all_q), bins=bin_edges)
                Pn, _ = np.histogram(np.concatenate(all_q), density=True, bins=bin_edges)
                avgQ = np.mean(np.concatenate(all_q))

                if not os.path.exists(savedir):
                    os.mkdir(savedir)
                os.chdir(savedir)
                np.save("n.npy", n)
                np.save("Pn.npy", Pn)
                np.save("bin_edges.npy", bin_edges)
                np.save("mid_bin.npy", mid_bin)
                np.savetxt("avg_Q.dat", np.array([avgQ]))

                if len(all_q) > 2:
                    dPn = np.std([ np.histogram(x, density=True, bins=bin_edges)[0] for x in all_q ], axis=0)
                    np.save("dPn.npy", dPn)
                os.chdir("..")
            else:
                print("No q.npy for this temperature: ", Tpaths[i]
                no_data = True

        if not no_data:
            os.chdir(savedir)
            plt.figure()
            if dPn is not None:
                plt.errorbar(mid_bin, Pn, yerr=dPn, lw=2)
            else:
                plt.plot(mid_bin, Pn, lw=2)
            plt.xlabel("Radius gyration (nm)")
            plt.ylabel("Prob density")
            plt.title(Tpaths[i] + r" $\langle R_g \rangle = {:.2f}$".format(avgQ))
            plt.savefig("Q_dist.pdf")
            plt.savefig("Q_dist.png")
            os.chdir("..")

        os.chdir(cwd)

