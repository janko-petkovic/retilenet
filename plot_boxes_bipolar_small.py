import os
import argparse

import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from modules.utils import Piper
from modules.models import DFC_LeNet_5, DFC_RetiNet, Deep_RetiNet


#----------------------------------------|
#            AUXILIARY METHODS           |
#----------------------------------------|


def ShiftMean(tensor, mu):
    copy = tensor.clone()
    return copy - mu

def ScaleVar(tensor, sigma):
    copy = tensor.clone()
    mu = copy.mean()
    return (copy - mu)/sigma + mu






#----------------------------------------|
#                  SETUP                 |
#----------------------------------------|


# --------------- PARSING ----------------
 
parser = argparse.ArgumentParser()

parser.add_argument("dataset_name",
                    help = "Name of the dataset [MNIST - FashionMNIST - SVHN]")

parser.add_argument("example_idx",
                    help = "Index of the chosen example",
                    type = int)

parser.add_argument(
    "depth",
    help="Retinic module depth",
    type=int
)

parser.add_argument("rks",
                    help="RetiNet retinic kernel size",
                    type = int)

args = parser.parse_args()

dataset_name = args.dataset_name
example_idx = args.example_idx
depth = args.depth
rks = args.rks



# ------ DEFINE THE RELATIVE PATHS --------------------------------------<<<<<<---------------

PATH_TO_DATASET = f"datasets/{dataset_name}"
PATH_TO_STATEDICTS = f"trained_models/{dataset_name}"
PATH_TO_SAVE = "small-figures"


# ------- DEFINE THE CHOSEN EXAMPLE ------
# ----------- FIND in_channels -----------

IL = { 
    "MNIST" : datasets.MNIST,
    "FashionMNIST" : datasets.FashionMNIST,
    "SVHN" : datasets.SVHN
}


if dataset_name == "SVHN":
    # pytorch does not use the same standards for all
    # datasets for some reasons I dont know why
    path = os.path.join(PATH_TO_DATASET, "SVHN")
    testset = IL[dataset_name](
        path,
        download = True,
        split = "test",
        transform = transforms.ToTensor()
    )

else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2)
    ])

    testset = IL[dataset_name](
        PATH_TO_DATASET,
        download = True,
        train = False,
        transform = transform
    )


example_tens = testset[example_idx][0]

print(example_tens.min(), example_tens.max(), example_tens.median())
in_channels = example_tens.shape[0]



# ------- MODEL DEFINITION AND HOOK SETUP ---------


device = "cuda" if torch.cuda.is_available() else "cpu"
retinet = Deep_RetiNet(depth = depth,
                       kernel_size = rks,
                       in_channels = in_channels).eval()

reti_state_dict = torch.load(
    f'{PATH_TO_STATEDICTS}/trained_Deep_RetiNet_d{depth}_rks{rks}_state_dict.pt',
    map_location = device)

retinet.load_state_dict(reti_state_dict)


# Attach a forward hook to the retinic layer of the retinet to get the
# first hidden output.
piper = Piper(retinet)
piper.define_subhooks(0,[0])







#----------------------------------------|
#                PLOTTING                |
#----------------------------------------|

# ----------- DATA GENERATION --------------
scale_hids = []
scale_origs = []
shift_hids = []
shift_origs = []

for i in [1,11,39]:
  var = i/10
  modified = ScaleVar(example_tens, var)

  retinet(modified.unsqueeze(axis=0).to(device))

  scale_origs.append(modified.reshape(1,-1).squeeze().numpy())
  scale_hids.append(piper.get_hidden_outputs()[0].reshape(1,-1).squeeze())


for i in [-20,0,20]:
  mean = i/10
  modified = ShiftMean(example_tens, mean)

  retinet(modified.unsqueeze(axis=0).to(device))

  shift_origs.append(modified.reshape(1,-1).squeeze().numpy())
  shift_hids.append(piper.get_hidden_outputs()[0].reshape(1,-1).squeeze())
  


# ---------- AESTHETICS SETUP -----------------
# Boxplot
median_width = 2
box_line_width = 0.8
whis_line_width = 0.8
cap_line_width = 0.8


# Outer chart features
axis_line_width = 0.5
grid_line_width = 0.3

tick_label_size = 10
tick_padding_size = 2

title_size = 15
title_pad = 10

x_label_size = 12
x_label_pad = 5

y_label_size = 12
y_label_pad = 5


# ----------- plotting ------------------
fig, axs = plt.subplots(2,2)
fig.set_size_inches(5.5,6)
fig.set_dpi(157)

fig.suptitle(f"{dataset_name}", size=20)

fig.tight_layout(
    rect=[0.07,0.05,1,0.93],
    w_pad = 0,
    h_pad = 2.5)

# Hashtable for subplot input and cosmetics
cfg = {
    "x" : [[shift_origs, shift_hids],
           [scale_origs, scale_hids]],

    "box_color" : ["black", "black"],

    "median_color" : ["red","teal"],

    "violin_color" : ["darkred", "teal"],


    "x_label" : [[r"$\mu$",r"$\mu$"],
                 [r"$\sigma$",r"$\sigma$"]],

    "y_lim" : [[-3,3], [-5,5]],

    # "x_ticks" : [[i for i in range(1,13,2)],
    #             [i for i in range(1,13,2)]],
    
    "x_tick_labels" : [[-2,0,2],
                       [0.1, 1.1, 3.9]]
}

for i, row in enumerate(axs):
    for j, ax in enumerate(row):

            bp = ax.violinplot(cfg["x"][i][j],
                                showmedians=False,
                                showextrema=False,
                                bw_method="scott")

            for body in bp["bodies"]:
                body.set_facecolor(cfg["violin_color"][j])
                body.set_edgecolor(cfg["violin_color"][j])
                body.set_linewidth(0.3)
            
            bp = ax.boxplot(cfg["x"][i][j], sym="",
                        whis=1.5, vert=True, widths=0.3,
                        showcaps=False, patch_artist=False
                        )

            for median in bp['medians']:
                median.set(color=cfg["median_color"][j], linewidth=median_width)

            for box in bp["boxes"]:
                box.set(linewidth=box_line_width, color=cfg["box_color"][j])


            for whis in bp["whiskers"]:
                whis.set(linewidth=whis_line_width)

            for cap in bp["caps"]:
                cap.set(linewidth=cap_line_width)

            for pos in ["top", "bottom", "right", "left"]:
                ax.spines[pos].set_linewidth(axis_line_width)

            ax.set_ylim(cfg["y_lim"][i])

            ax.set_xlabel(cfg["x_label"][i][j], fontsize = x_label_size, labelpad=x_label_pad)
            if not j:
                ax.set_ylabel("Pixel values", fontsize=y_label_size, labelpad=y_label_pad)

            # ax.set_xticks(cfg["x_ticks"][i])
            ax.set_xticklabels(cfg["x_tick_labels"][i], fontsize = tick_label_size)

            ax.tick_params(axis='x', labelsize=tick_label_size, pad=tick_padding_size)
            ax.tick_params(axis='y', labelsize=tick_label_size, pad=tick_padding_size)

            if j==1:
                ax.tick_params(left=False, labelleft=False)

            ax.grid(axis="y", which='major', color='#888888', linestyle=':', linewidth = grid_line_width)


axs[0,0].set_title(f"Input", fontsize=title_size, pad=title_pad)
axs[0,1].set_title(f"First HO", fontsize=title_size, pad=title_pad)


# Save and show
PATH_TO_SAVE = os.path.join(
    PATH_TO_SAVE, f"boxes_retinet_d{depth}_rks{rks}_{dataset_name}_bipolar"
)
plt.savefig(PATH_TO_SAVE)

plt.show()