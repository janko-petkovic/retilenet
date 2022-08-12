###############
# PROLEGOMENA #
###############


import os, sys
import torch
torch.manual_seed(42)

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
import pandas as pd

from numpy.random import default_rng

from tqdm import tqdm

from torchvision import datasets, transforms

from PIL import Image


PATH_TO_ROOT = "."
PATH_TO_DATASETS = os.path.join(PATH_TO_ROOT, "datasets")
PATH_TO_STATE_DICTS = os.path.join(PATH_TO_ROOT, "trained_models")

from modules.utils import Trainer
from modules.models import DFC_LeNet_5, Deep_RetiNet, BNDFC_LeNet_5, INDFC_LeNet_5


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device}')



################################
# TESTSET TRANSFORMING METHODS #
################################

class ShiftMean(object):
  def __init__(self, mu):
    self.mu = mu
  
  def collate(self, batch):
    imgs = []
    labels = []

    for pair in batch:
      shifted = pair[0] - self.mu
      imgs.append(shifted)
      labels.append(pair[1])

    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels)

    return [imgs, labels]


class ScaleVar(object):
  def __init__(self, sigma):
    self.sigma = sigma
  
  def collate(self, batch):
    imgs = []
    labels = []

    for pair in batch:
      mean = pair[0].mean()
      scaled = (pair[0]-mean)/self.sigma + mean

      imgs.append(scaled)
      labels.append(pair[1])

    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels)

    return [imgs, labels]


##########
# SWEEPS #
##########


############## - Input panel - ###############
# MNIST - FashionMNIST - SVHN
dataset_name = sys.argv[1]

# Deep_RetiNet params
retinic_kernel_size =  7
depth = 3 
rks = retinic_kernel_size

#############################################3



# Testset loading

IL = { 
    "MNIST" : datasets.MNIST,
    "FashionMNIST" : datasets.FashionMNIST,
    "SVHN" : datasets.SVHN
}


if dataset_name == "SVHN":
    # pytorch does not use the same standards for all
    # datasets for some reasons I dont know why
    path = os.path.join(PATH_TO_DATASETS, "SVHN")

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
        PATH_TO_DATASETS,
        download = True,
        train = False,
        transform = transform
    )
 
in_channels = testset[0][0].shape[0]
test_size = testset.__len__()


# Model instantiation and state_dict loading
retinet_path = os.path.join(
    PATH_TO_STATE_DICTS, dataset_name, f"trained_Deep_RetiNet_d{depth}_rks{rks}_state_dict.pt") 

lenet_path = os.path.join(
    PATH_TO_STATE_DICTS, dataset_name, f"trained_DFC_LeNet_5_state_dict.pt")

bnlenet_path = os.path.join(
    PATH_TO_STATE_DICTS, dataset_name, f"trained_BNDFC_LeNet_5_state_dict.pt")

inlenet_path = os.path.join(
    PATH_TO_STATE_DICTS, dataset_name, f"trained_INDFC_LeNet_5_state_dict.pt")

retinet = Deep_RetiNet(depth, rks, in_channels).to(device).eval()
retinet.load_state_dict(torch.load(retinet_path, map_location = device))

lenet = DFC_LeNet_5(in_channels).to(device).eval()
lenet.load_state_dict(torch.load(lenet_path, map_location = device))

bnlenet = BNDFC_LeNet_5(in_channels).to(device).eval()
bnlenet.load_state_dict(torch.load(bnlenet_path, map_location = device))

inlenet = INDFC_LeNet_5(in_channels).to(device).eval()
inlenet.load_state_dict(torch.load(inlenet_path, map_location = device))



# Sweep loops

# Mean sweep loop:   mean = i/10

results_mean = []

for i in tqdm(range(-20,21,2)):
  mean = i/10

  transform = ShiftMean(mean)

  testloader = DataLoader(
      testset, batch_size = test_size,
      collate_fn = transform.collate)

  for batch, labels in testloader:
    batch, labels = batch.to(device), labels.to(device)

    out_lenet = lenet(batch)
    out_bnlenet = bnlenet(batch)
    out_inlenet = inlenet(batch)
    out_retinet = retinet(batch)
    

    acc_lenet = (labels == out_lenet.argmax(axis=1)).sum()/test_size
    acc_bnlenet = (labels == out_bnlenet.argmax(axis=1)).sum()/test_size
    acc_inlenet = (labels == out_inlenet.argmax(axis=1)).sum()/test_size
    acc_retinet = (labels == out_retinet.argmax(axis=1)).sum()/test_size
  
  results_mean.append([mean, acc_lenet.item(), acc_bnlenet.item(), acc_inlenet.item(), acc_retinet.item()])


# Deviation sweep loop:    deviation = i/10

results_var = []

for i in tqdm(range(1,41,2)):
  var = i/10

  transform = ScaleVar(var)

  testloader = DataLoader(
      testset, batch_size = test_size,
      collate_fn = transform.collate)

  for batch, labels in testloader:
    batch, labels = batch.to(device), labels.to(device)

    out_lenet = lenet(batch)
    out_retinet = retinet(batch)
    out_bnlenet = bnlenet(batch)
    out_inlenet = inlenet(batch)

    acc_lenet = (labels == out_lenet.argmax(axis=1)).sum()/test_size
    acc_retinet = (labels == out_retinet.argmax(axis=1)).sum()/test_size
    acc_bnlenet = (labels == out_bnlenet.argmax(axis=1)).sum()/test_size
    acc_inlenet = (labels == out_inlenet.argmax(axis=1)).sum()/test_size
  
  results_var.append([var, acc_lenet.item(), acc_bnlenet.item(), acc_inlenet.item(), acc_retinet.item()])



# Results preview
printable_mean = np.stack(results_mean, axis=0)
printable_var = np.stack(results_var, axis=0)

fig = plt.figure(figsize=(18,5))
tight = fig.tight_layout(w_pad=40)
fig.set_tight_layout(tight)

fig1 = fig.add_subplot(1,2,1)
fig1.set_title("Accuracy versus $\mu$ sweep", size=20, pad=20)

fig1.set_xlabel("$\mu$", fontsize=15, labelpad=15)
fig1.set_ylim([0,1])
fig1.set_ylabel("Accuracy", fontsize=15, labelpad=15)

fig1.tick_params(axis='x', labelsize=15, pad=10)
fig1.tick_params(axis='y', labelsize=15, pad=10)

plt.scatter(printable_mean[:,0], printable_mean[:,1], 
            marker='o', label=f'LeNet_5')

plt.scatter(printable_mean[:,0], printable_mean[:,2], 
            marker='o', label=f'BNLeNet_5')

plt.scatter(printable_mean[:,0], printable_mean[:,3], 
            marker='o', label=f'INLeNet_5')

plt.scatter(printable_mean[:,0], printable_mean[:,4],
            marker='o', label=f'Deep_RetiNet_d{depth}_rks{rks}')

plt.legend(loc=3, fontsize=15)

fig2 = fig.add_subplot(1,2,2)
fig2.set_title("Accuracy versus $\sigma$ sweep", fontsize = 20, pad=20)
fig2.set_xlabel("$\sigma$", fontsize = 15, labelpad=15)
fig2.set_ylim([0,1])
fig2.set_ylabel("Accuracy", fontsize = 15, labelpad=15)


fig2.tick_params(axis='x', labelsize=15, pad=10)
fig2.tick_params(axis='y', labelsize=15, pad=10)

plt.scatter(printable_var[:,0], printable_var[:,1], 
            marker='o', label='LeNet_5')

plt.scatter(printable_var[:,0], printable_var[:,2], 
            marker='o', label='BNLeNet_5')

plt.scatter(printable_var[:,0], printable_var[:,3], 
            marker='o', label='INLeNet_5')

plt.scatter(printable_var[:,0], printable_var[:,4], 
            marker='o', label=f'Deep_RetiNet_d{depth}_rks{rks}')

plt.legend(loc=3, fontsize=15)

plt.show()



# Output .csv generation
df_results_mean = pd.DataFrame(elem for elem in results_mean)
df_results_var = pd.DataFrame(elem for elem in results_var)


df_results_mean.rename(
    columns = {0 : "mean_off", 
               1 : "acc_lenet", 
               2 : "acc_bnlenet",
               3 : "acc_inlenet",
               4 : "acc_retinet"},
    inplace = True)

df_results_var.rename(
    columns = {0 : "var_resc", 
               1 : "acc_lenet", 
               2 : "acc_bnlenet",
               3 : "acc_inlenet",
               4 : "acc_retinet"},
    inplace = True)



mean_file_name = f"accuracy_sweep_Deep_RetiNet_d{depth}_rks{rks}_vs_mean.csv"
PATH_TO_SAVE_FILE = os.path.join(
    PATH_TO_ROOT, f"results/accuracy_sweeps/{dataset_name}", mean_file_name)

df_results_mean.to_csv(PATH_TO_SAVE_FILE, index=False)


var_file_name = f"accuracy_sweep_Deep_RetiNet_d{depth}_rks{rks}_vs_var.csv"
PATH_TO_SAVE_FILE = os.path.join(
    PATH_TO_ROOT, f"results/accuracy_sweeps/{dataset_name}", var_file_name)

df_results_var.to_csv(PATH_TO_SAVE_FILE, index=False)

