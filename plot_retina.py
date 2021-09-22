import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from modules.models import Deep_RetiNet


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
    "depth",
    help="The retinic kernel size of the retinet to be used",
    type=int
)

parser.add_argument(
    "retinic_kernel_size",
    help="The retinic kernel size of the retinet to be used",
    type=int
)

args = parser.parse_args()

dataset_name = args.dataset_name
depth = args.depth
kernel_size = args.retinic_kernel_size


# IF YOU MOVE THE SCRIPT CHANGE ONLY THIS WITH NEW RELATIVE PATHS  <-------------------------------------------------------!!!
LOAD_PATH = f"trained_models/{dataset_name}/trained_Deep_RetiNet_d{depth}_rks{kernel_size}_state_dict.pt"
SAVE_PATH = f"plots/accuracy_sweeps/retina/{dataset_name}_Deep_RetiNet_d{depth}_rks{kernel_size}_retina.png"


# this line is pretty wrong
in_channels = 3 if dataset_name=="SVHN" else 1


device = "cuda" if torch.cuda.is_available() else "cpu"
model = Deep_RetiNet(depth, kernel_size, in_channels)
model.load_state_dict(torch.load(LOAD_PATH, map_location=device))



retinas = []
# this implementation is pretty sloppy as I already know that the 
# filters are every third position. Would be better to identify them
# first
for i, child in enumerate(model.retina.children()):
    if not (i % 3): retinas.append(child.weight.detach().numpy())

# this is just the number of filters, 3 or 1
n_filters = retinas[0].shape[0]



#----------------------------------------|
#                PLOTTING                |
#----------------------------------------|


fig = plt.figure(figsize=(5,5*depth))
fig.set_dpi(157)

fig.tight_layout(
    rect=[0,0,0.95,0.95],
    w_pad = 10,
    h_pad = 0)


plt.suptitle(f"{dataset_name} Deep_RetiNet_d{depth}_ks{kernel_size}")

for i, cell in enumerate(retinas):
    for j, row in enumerate(cell):
        for k, filter in enumerate(row):
            fig.add_subplot(n_filters*depth, n_filters,
                k+1 + j*n_filters + i*n_filters**2)
            plt.imshow(filter, cmap="gray")
            
            plt.xticks([])
            plt.yticks([])

            cbar = plt.colorbar(shrink=0.8)
            cbar.ax.tick_params(labelsize=6) 

plt.show()