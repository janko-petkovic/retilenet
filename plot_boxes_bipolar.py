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
    help="Number of precortical conv-drop-tanh blocks",
    type=int
)

parser.add_argument(
    "kernel_size",
    help="Kernel size of the precortical convolutional layers",
    type=int
)

parser.add_argument(
    "augmented",
    help="If the model has been trained using data augmentation",
    type=bool
)

args = parser.parse_args()

dataset_name = args.dataset_name
example_idx = args.example_idx
depth = args.depth
rks = args.kernel_size
augmented = args.augmented


# ------ DEFINE THE RELATIVE PATHS --------------------------------------<<<<<<---------------

PATH_TO_DATASET = f"datasets"
PATH_TO_STATEDICTS = f"trained_models/{dataset_name}"
PATH_TO_SAVE = "plots/box_plots"


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
        transforms.Pad(2),
        transforms.ToTensor()
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
                       in_channels = in_channels).to(device).eval()

if augmented:
    reti_state_dict = torch.load(
        f'{PATH_TO_STATEDICTS}/trained_Deep_RetiNet_d{depth}_rks{rks}_augmented_state_dict.pt',
        map_location = device)
else:
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
x_shift = []
x_scale = []

for i in range(1,42,10):
  var = i/10
  x_scale.append(var)
  modified = ScaleVar(example_tens, var)

  retinet(modified.unsqueeze(axis=0).to(device))

  scale_origs.append(modified.reshape(1,-1).squeeze().numpy())
  scale_hids.append(piper.get_hidden_outputs()[0].reshape(1,-1).squeeze())


for i in range(-20,21,10):
  mean = i/10
  x_shift.append(mean)
  modified = ShiftMean(example_tens, mean)

  retinet(modified.unsqueeze(axis=0).to(device))

  shift_origs.append(modified.reshape(1,-1).squeeze().numpy())
  shift_hids.append(piper.get_hidden_outputs()[0].reshape(1,-1).squeeze())
  
x_shift = np.array(x_shift)
x_scale = np.array(x_scale)

# Second plotting 
# The data is shift_hids, shift_origs etc

fig, axs = plt.subplots(2,1, figsize=(4,5), dpi=200)
fig.tight_layout(pad=2)

orig_style = {
        'showcaps' : False,
        'boxprops' : {'color' : 'tab:red', 'facecolor' : 'white', 'linewidth' : 1.5},
        'whiskerprops' : {'color' : 'tab:red', 'linewidth' : 1.5},
        'medianprops' : {'color' : 'tab:red', 'linewidth' : 3}
}

hid_style = {
        'showcaps' : False,
        'boxprops' : {'color' : 'tab:green', 'facecolor' : 'white', 'linewidth' : 1.5},
        'whiskerprops' : {'color' : 'tab:green', 'linewidth' : 1.5},
        'medianprops' : {'color' : 'tab:green', 'linewidth' : 3},
}



ax = axs[0]
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
bp1 = ax.boxplot(shift_origs, sym='', whis=[0,100], positions=x_shift-0.15, widths=0.2, showfliers=True, 
                patch_artist=True, **orig_style)
bp2 = ax.boxplot(shift_hids, sym='', positions=x_shift+0.15, widths=0.2, showfliers=True,
                patch_artist=True, **hid_style)
ax.set_xlabel(r'$\mu$')
ax.set_xticks(x_shift)
ax.set_xticklabels(x_shift)
ax.set_ylim(-3,4.2)
ax.set_title(dataset_name)
ax.legend([bp1['medians'][0], bp2['medians'][0]], ['example', 'hidden out'], loc='upper right', ncol=2)



ax = axs[1]
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
bp1 = ax.boxplot(scale_origs, sym='', whis=[0,100], positions=x_shift-0.15, widths=0.2, showfliers=True, 
                patch_artist=True, **orig_style)
bp2 = ax.boxplot(scale_hids, sym='', positions=x_shift+0.15, widths=0.2, showfliers=True,
                patch_artist=True, **hid_style)
ax.set_xlabel(r'$\sigma$')
ax.set_xticks(x_shift)
ax.set_xticklabels(x_scale)
ax.set_ylim(-15,15)
ax.legend([bp1['medians'][0], bp2['medians'][0]], ['example', 'hidden out'], loc='upper right', ncol=2)



PATH_TO_SAVE = os.path.join(
    PATH_TO_SAVE, f"boxes_retinet_d{depth}_rks{rks}_{dataset_name}_bipolar.png"
)

plt.savefig(PATH_TO_SAVE)
plt.show()
