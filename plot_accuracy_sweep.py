import os, re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#----------------------------------
#              SETUP              |
#---------------------------------- 


# Parsing: dataset_name, retinic_kernel_size
parser = argparse.ArgumentParser()

parser.add_argument(
    "dataset_name",
    help="Name of the dataset on which the accuracy sweep has to be carried out",
    type=str
)

parser.add_argument(
    "depth",
    help="Retinic module depth",
    type=int
)

parser.add_argument(
    "retinic_kernel_size",
    help="Retinic kernel size of the retinet to be used",
    type=int
)

args = parser.parse_args()

dataset_name = args.dataset_name
depth = args.depth
kernel_size = args.retinic_kernel_size



# IF YOU MOVE THE SCRIPT CHANGE ONLY THIS WITH NEW RELATIVE PATHS  <----------------------------------!!!
LOAD_PATH = f"results/accuracy_sweeps/{dataset_name}"
SAVE_PATH = "plots/accuracy_sweeps"




# Import data: csv -> dataframe -> array
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
    rect=[0.03, 0.1, 1, 0.9],
    w_pad = 8,
    h_pad = 5)


for i, ax in enumerate(axs):
    xyy = ax_config["data"][i]

    # scatter the two charts
    ax.scatter(xyy[:,0],xyy[:,1],
        marker = 'o', facecolors="", edgecolors="black", label = "LeNet_5")

    ax.scatter(xyy[:,0], xyy[:,2], 
        marker = 'x', c="r", s=60, label = f"RetiLeNet")
    
    # y limits
    ax.set_ylim([0,1])

    # subplot title
    ax.set_title(ax_config["title"][i], size = 18, pad = 20)

    # tick dimension and padding
    ax.tick_params(axis='x', labelsize=13, pad=10)
    ax.tick_params(axis='y', labelsize=13, pad=10)

    # axes labels
    ax.set_xlabel(ax_config["xlabel"][i], fontsize=15, labelpad=15)
    ax.set_ylabel("Accuracy", fontsize=15, labelpad=15)
    
    ax.legend(loc=6, fontsize=15)


# Save the figure and show preview
SAVE_PATH = os.path.join(
    SAVE_PATH, 
    f"accuracy_sweep_retinet_d{depth}_rks{kernel_size}_{dataset_name}.png"
)

plt.savefig(SAVE_PATH)

plt.show()