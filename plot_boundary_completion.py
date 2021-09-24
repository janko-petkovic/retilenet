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


class HiddenViewer():
  def __init__(self, net):
    self.net = net
    self.piper = Piper(net)
    self.piper.define_hooks([0,1])

  def view(self, tens):
    input = tens.reshape(28,28)
    out = self.net(input)

    guess = out.argmax()

    hiddens = self.piper.get_hidden_outputs()
    s_hiddens = np.array(hiddens[0].squeeze())
    c_hiddens = np.array(hiddens[1]).squeeze()
    
    s_max = s_hiddens.max(axis=0)
    c_max = c_hiddens.max(axis=0)

    s_argmax = s_hiddens.argmax(axis=0)
    c_argmax = c_hiddens.argmax(axis=0)

    return [s_max, s_argmax,
            c_max, c_argmax,
            guess]




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
PATH_TO_SAVE = "plots/boundary_completions"




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
example_label = testset[example_idx][1]

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


# Attach a forward hook to the last retinic layer, and to the 
# two lenet convs
piper = Piper(retinet)
piper.define_subhooks(0,[6])
piper.define_subhooks(1,[1,3])



# #----------------------------------------|
# #                PLOTTING                |
# #----------------------------------------|


# # ------------ DATA GENERATION -------------------

predictions = []
plot_table = []

for i in range(5):
    
    example_tens[:,16-i:16+i,:] = 0
    out = retinet(example_tens.reshape(1,in_channels,32,32))
    prediction = out.argmax()

    hiddens = piper.get_hidden_outputs()

    retina_hiddens = np.stack(hiddens[:in_channels], axis=0)
    simple_hiddens = np.stack(hiddens[in_channels:in_channels+6], axis=0)
    complex_hiddens = np.stack(hiddens[in_channels+6:], axis=0) 

    simple_max = simple_hiddens.max(axis=0)
    simple_args = simple_hiddens.argmax(axis=0)

    complex_max = complex_hiddens.max(axis=0)
    complex_args = complex_hiddens.argmax(axis=0)

    # plot_table.append([
    #     example_tens.permute(1,2,0).clone(),
    #     retina_hiddens.transpose(1,2,0)/retina_hiddens.max(),
    #     simple_max,
    #     simple_args,
    #     complex_max,
    #     complex_args])

    plot_table.append(hiddens[in_channels+6:])

    predictions.append(prediction)


fig, axes = plt.subplots(5,16)
fig.tight_layout()
fig.set_size_inches(30,15)

for i in range(5):
    for j in range(16):
        axes[i,j].imshow(plot_table[i][j])
        axes[i,j].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

print(predictions)

plt.show()


# # ---------- AESTHETICS SETUP -----------------
# # Boxplot
# median_width = 2
# box_line_width = 0.8
# whis_line_width = 0.8
# cap_line_width = 0.8


# # Outer chart features
# axis_line_width = 0.5
# grid_line_width = 0.3

# tick_label_size = 8
# tick_padding_size = 2

# title_size = 13
# title_pad = 15

# x_label_size = 12
# x_label_pad = 5

# y_label_size = 12
# y_label_pad = 5


# # ----------- plotting ------------------
# fig, axs = plt.subplots(2,2)
# fig.set_size_inches(10,6)
# fig.set_dpi(157)

# fig.tight_layout(
#     rect=[0.05,0.05,0.95,0.95],
#     w_pad = 5,
#     h_pad = 4)

# # Hashtable for subplot input and cosmetics
# cfg = {
#     "x" : [[shift_origs, shift_hids],
#            [scale_origs, scale_hids]],

#     "box_color" : ["black", "black"],

#     "median_color" : ["red","teal"],

#     "violin_color" : ["darkred", "teal"],


#     "x_label" : [[r"$\mu$",r"$\mu$"],
#                  [r"$\sigma$",r"$\sigma$"]],

#     "y_lim" : [[-3,3], [-5,5]],

#     "x_ticks" : [[i for i in range(1,13,2)],
#                 [i for i in range(1,13,2)]],
    
#     "x_tick_labels" : [[i/10 for i in range(-20,21,8)],
#                        [i/10 for i in range(1,43,8)]]
# }

# for i, row in enumerate(axs):
#     for j, ax in enumerate(row):

#             bp = ax.violinplot(cfg["x"][i][j],
#                                 showmedians=False,
#                                 showextrema=False,
#                                 bw_method="scott")

#             for body in bp["bodies"]:
#                 body.set_facecolor(cfg["violin_color"][j])
#                 body.set_edgecolor(cfg["violin_color"][j])
#                 body.set_linewidth(0.3)
            
#             bp = ax.boxplot(cfg["x"][i][j], sym="",
#                         whis=1.5, vert=True, widths=0.3,
#                         showcaps=False, patch_artist=False
#                         )

#             for median in bp['medians']:
#                 median.set(color=cfg["median_color"][j], linewidth=median_width)

#             for box in bp["boxes"]:
#                 box.set(linewidth=box_line_width, color=cfg["box_color"][j])


#             for whis in bp["whiskers"]:
#                 whis.set(linewidth=whis_line_width)

#             for cap in bp["caps"]:
#                 cap.set(linewidth=cap_line_width)

#             for pos in ["top", "bottom", "right", "left"]:
#                 ax.spines[pos].set_linewidth(axis_line_width)

#             ax.set_ylim(cfg["y_lim"][i])

#             ax.set_xlabel(cfg["x_label"][i][j], fontsize = x_label_size, labelpad=x_label_pad)
#             ax.set_ylabel("Pixel values", fontsize=y_label_size, labelpad=y_label_pad)

#             ax.set_xticks(cfg["x_ticks"][i])
#             ax.set_xticklabels(cfg["x_tick_labels"][i], fontsize = tick_label_size)

#             ax.tick_params(axis='x', labelsize=tick_label_size, pad=tick_padding_size)
#             ax.tick_params(axis='y', labelsize=tick_label_size, pad=tick_padding_size)

#             ax.grid(axis="y", which='major', color='#888888', linestyle=':', linewidth = grid_line_width)


# axs[0,0].set_title(f"{dataset_name} example", fontsize=title_size, pad=title_pad)
# axs[0,1].set_title(f"Bipolar cell hidden output", fontsize=title_size, pad=title_pad)


# # Save and show
# PATH_TO_SAVE = os.path.join(
#     PATH_TO_SAVE, f"boxes_retinet_d{depth}_rks{rks}_{dataset_name}_bipolar"
# )
# plt.savefig(PATH_TO_SAVE)

# plt.show()