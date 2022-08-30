import os, re, sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#----------------------------------
#              SETUP              |
#---------------------------------- 


# Parsing: dataset_name, retinic_kernel_size
# Example: $ python plot_accuracy_sweep.py MNIST True
dataset_name = sys.argv[1]
large_sweep = (sys.argv[2] == 'True')

depth = 3 #args.depth
kernel_size = 7 #args.kernel_size


# IF YOU MOVE THE SCRIPT CHANGE ONLY THIS WITH NEW RELATIVE PATHS  <----------------------------------!!!
LOAD_PATH = f"results/accuracy_sweeps/{dataset_name}"
SAVE_PATH = "plots/accuracy_sweeps"




# Import data: csv -> dataframe -> array
if large_sweep:
    mean_path = os.path.join(
        LOAD_PATH, f"large_accuracy_sweep_Deep_RetiNet_d{depth}_rks{kernel_size}_vs_mean.csv")
    var_path = os.path.join(
        LOAD_PATH, f"large_accuracy_sweep_Deep_RetiNet_d{depth}_rks{kernel_size}_vs_var.csv")
else:
    mean_path = os.path.join(
        LOAD_PATH, f"accuracy_sweep_Deep_RetiNet_d{depth}_rks{kernel_size}_vs_mean.csv")
    var_path = os.path.join(
        LOAD_PATH, f"accuracy_sweep_Deep_RetiNet_d{depth}_rks{kernel_size}_vs_var.csv")

results_mean = pd.read_csv(mean_path)
results_var = pd.read_csv(var_path)

printable_mean = np.array(results_mean)
printable_var = np.array(results_var)




#----------------------------------
#             PLOTTING            |
#---------------------------------- 


# Chart config table

ax_config = {
    "data" : [
        printable_mean,
        printable_var
    ],
    "title" : [
        f"{dataset_name} accuracy versus $\mu$ sweep",
        f"{dataset_name} accuracy versus $\sigma$ sweep"
    ],

    "xlabel" : [
        "$\mu$",
        "$\sigma$"
    ]
}



# Plotting loop

fig, axs = plt.subplots(1,2)
fig.set_size_inches(15,5)

fig.tight_layout(
    rect=[0.08, 0.1, 1, 0.9],
    w_pad = 8,
    h_pad = 5)


# For the large plots
if large_sweep:
    axs[0].set_ylim(0.8,1)
    axs[1].set_ylim(0.0,1)

    axs[0].axvspan(-2,2, alpha=0.3, color='gray', label='training range')
    axs[1].axvspan(0.1,4, alpha=0.3, color='gray', label='training range')


for i, ax in enumerate(axs):
    xyy = ax_config["data"][i]
    # scatter the two charts
    # ax.scatter(xyy[:,0],xyy[:,1],
    #     marker = 'o', facecolors="white", edgecolors="black", label = "LeNet_5")


    # ax.plot(xyy[:,0],xyy[:,1],
    #     marker = '',  label = "LeNet_5")#color='#cfe2f2',

    # ax.plot(xyy[:,0],xyy[:,3],
    #    marker = '',  label = "BNLeNet_5")#color= '#f5cdad',
    # # 
    # # # ax.plot(xyy[:,0],xyy[:,5],
    # # #     marker = 'x', label = "LeNet_5 + IN")

    # ax.plot(xyy[:,0], xyy[:,6], 
    #     marker = 'o',  linewidth=2, mfc='white', mew=3, label = f"RetiLeNet")#color='#b8e5a9',
    
    ax.plot(xyy[:,0],xyy[:,2],
        marker = '', label = "LeNet_5 + DA")

    ax.plot(xyy[:,0],xyy[:,4],
        marker = '', label = "BNLeNet_5 + DA")

    ax.plot(xyy[:,0], xyy[:,7], 
        marker = 'o', linewidth=3, mfc='white', mew=3, label = f"RetiLeNet + DA")
    
    if not large_sweep:
        ax.set_ylim([0,1])

    # subplot title
    ax.set_title(ax_config["title"][i], size = 18, pad = 20)

    # tick dimension and padding
    ax.tick_params(axis='x', labelsize=13, pad=10)
    ax.tick_params(axis='y', labelsize=13, pad=10)

    # axes labels
    ax.set_xlabel(ax_config["xlabel"][i], fontsize=15, labelpad=15)
    ax.set_ylabel("Accuracy", fontsize=15, labelpad=15)
    
    if i==0:
        ax.legend(loc='upper left', fontsize=15)
    else:
        ax.legend(loc='lower left', fontsize=15)


# Save the figure and show preview
if large_sweep:
    SAVE_PATH = os.path.join(
        SAVE_PATH, 
        f"large_accuracy_sweep_retinet_d{depth}_rks{kernel_size}_{dataset_name}.png"
    )
else:
    SAVE_PATH = os.path.join(
        SAVE_PATH, 
        f"accuracy_sweep_retinet_d{depth}_rks{kernel_size}_{dataset_name}.png"
    )

plt.savefig(SAVE_PATH)

plt.show()
