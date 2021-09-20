import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from modules.models import DFC_RetiNet


#----------------------------------
#              SETUP              |
#---------------------------------- 


# Parsing: dataset_name, retinic_kernel_size
parser = argparse.ArgumentParser()

parser.add_argument(
    "dataset_name",
    help="The name of the dataset on which the accuracy sweep has to be carried out",
    type=str
)

parser.add_argument(
    "retinic_kernel_size",
    help="The retinic kernel size of the retinet to be used",
    type=int
)

args = parser.parse_args()

dataset_name = args.dataset_name
kernel_size = args.retinic_kernel_size


# IF YOU MOVE THE SCRIPT CHANGE ONLY THIS WITH NEW RELATIVE PATHS  <----------------------------------!!!
LOAD_PATH = f"trained_models/{dataset_name}/trained_DFC_RetiNet_{kernel_size}_state_dict.pt"
SAVE_PATH = f"plots/accuracy_sweeps/retina/{dataset_name}_DFC_RetiNet_{kernel_size}_retina.png"

# this line is pretty wrong
in_channels = 3 if dataset_name=="SVHN" else 1

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DFC_RetiNet(kernel_size, in_channels)
model.load_state_dict(torch.load(LOAD_PATH, map_location=device))

retina = model.retina.weight.detach().numpy()

n = retina.shape[0]

fig = plt.figure(figsize=(7,5))
fig.set_dpi(157)

fig.tight_layout(
    rect=[0,0.05,0.95,0.95],
    w_pad = 20,
    h_pad = 0)


plt.suptitle(f"{dataset_name} RetiNet {kernel_size}")

for i, row in enumerate(retina):
    for j, filter in enumerate(row):
        fig.add_subplot(n,n,j+1+i*n)
        plt.imshow(filter, cmap="gray")
        
        plt.xticks([])
        plt.yticks([])

        cbar = plt.colorbar(shrink=0.8)
        cbar.ax.tick_params(labelsize=6) 

plt.show()