import os
import glob

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #savedir = "coeff_vs_s_gauss"
    #savedir = "coeff_vs_s_bond_angle_wca_gauss"
    n_beads = 25

    bonds = True
    angles = True
    non_bond_wca = True

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

    savedir += dirsuff

    os.chdir(savedir)

    line_colors = ["#ff7f00", "#377eb8", "grey", "#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#e41a1c", "#4daf4a", "#984ea3"] # orange, blue, grey

    dt_frame = 0.002
    gamma = 100

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

    ylabels = [r"$k_b$", r"$k_a$", r"$\epsilon$"]
    coeff_true = [100, 20, 1]

    plt.figure()
    for i in range(start_k):

        for j in range(len(s_list)):
            coeff = gamma*c_vs_s[i][j]
            coeff_ratio = coeff/coeff_true[i]

            #c_avg = np.mean(coeff, axis=1)
            #c_avg_err = np.std(coeff, axis=1)/np.sqrt(float(coeff.shape[1]))
            #c_avg /= coeff_true[i]
            #c_avg_err /= coeff_true[i]
            #c_avg *= 40
            #c_avg_err *= 40

            #plt.errorbar(dt_frame, c_avg, yerr=c_avg_err, color=line_colors[i], label=ylabels[i])
            for n in range(len(coeff_ratio)):
                if n == 0 and j == 0:
                    plt.plot([dt_frame*s_list[j]], [coeff_ratio[n]], 'o', ms=8, color=line_colors[i], label=ylabels[i])
                else:
                    plt.plot([dt_frame*s_list[j]], [coeff_ratio[n]], 'o', ms=8, color=line_colors[i])

    #plt.legend(loc=1, fontsize=20, fancybox=False, frameon=True, edgecolor="k", framealpha=1)
    legend = plt.legend(loc=1, fontsize=20, fancybox=False, frameon=True)
    legend.get_frame().set_edgecolor("k") 
    plt.ylabel("Effective / True", fontsize=26)
    plt.xlim(0.9*dt_frame, 1.1*dt_frame*np.max(s_list))
    plt.semilogx()
    plt.xlabel(r"$\Delta t$ (ps)", fontsize=26)
    plt.savefig("coeff_vs_s_3_params_lstsq_ratio_xlog_indv.pdf")
    plt.savefig("coeff_vs_s_3_params_lstsq_ratio_xlog_indv.png")
    #plt.savefig("coeff_vs_s_3_params_lstsq_xlog_with_true.pdf")
    #plt.savefig("coeff_vs_s_3_params_lstsq_xlog_with_true.png")
