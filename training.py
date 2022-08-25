'''
Script adapted from training.ipynb to train LeNet 5 with different augmentation
strategies, namely:
    - classic random augmentation
    - batch normalization
    - instance normalization

For batch and instance normalization we have to implement two additional lenets,
namely:
    - dfclenet5_batch.py
    - dfclenet5_inst.py

For the classic data augmentation we only implement dfclenet5
'''

###############
# PROLEGOMENA #
###############

# Imports

import os, sys
import torch
torch.manual_seed(42)

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from tqdm import tqdm

from torchvision import datasets, transforms

import gc



# PATH, custom imports, device definition

PATH_TO_ROOT = "."
PATH_TO_DATASETS = os.path.join(
    PATH_TO_ROOT, "datasets"
)

from modules.utils import Trainer
from modules.models import DFC_LeNet_5, Deep_RetiNet, BNDFC_LeNet_5, INDFC_LeNet_5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device}')



############
# TRAINING #
############

#################### - interface - #########################
# Get an argv handle for both dataset and neural network
# MNIST - FashionMNIST - SVHN
dataset_name = sys.argv[1] # "MNIST"         

# LeNet - Deep_RetiNet - BNLeNet - INLeNet
model_name = sys.argv[2] # "Deep_RetiNet"

# Parameters for Deep_RetiNet
retinic_kernel_size = 7
rks = retinic_kernel_size
depth =  3
############################################################


# Further definitions:
# optimizer, batch_size, lr, epochs, loss_fn, dataset choice
optimizer = Adam 
batch_size = 128 
start_lr = 1e-3 
frac_epochs = 5

loss_fn = CrossEntropyLoss()


IL = { 
    "MNIST" : datasets.MNIST,
    "FashionMNIST" : datasets.FashionMNIST,
    "SVHN" : datasets.SVHN,
}


if dataset_name == "SVHN":
    # pytorch does not use the same standards for all
    # datasets for some reasons I dont know why
    path = os.path.join(PATH_TO_DATASETS, "SVHN")
    transform = transforms.ToTensor()

    trainset = IL[dataset_name](
        path,
        download = True,
        split = "train",
        transform = transform
    )

    testset = IL[dataset_name](
        path,
        download = True,
        split = "test",
        transform = transform
    )

else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2)
    ])

    trainset = IL[dataset_name](
        PATH_TO_DATASETS,
        download = True,
        train = True,
        transform = transform
    )

    testset = IL[dataset_name](
        PATH_TO_DATASETS,
        download = True,
        train = False,
        transform = transform
    )
 
in_channels = testset[0][0].shape[0]
train_size = trainset.__len__()
test_size = testset.__len__()


trainloader = DataLoader(trainset, 
                          batch_size = batch_size, 
                          shuffle = True, 
                          num_workers = 1)

testloader = DataLoader(testset, 
                        batch_size = 5000, 
                        shuffle = False, 
                        num_workers = 1)


# Model and optimizer istantiation
if model_name == "LeNet":
  model = DFC_LeNet_5(in_channels).to(device)
  model_save_name = "DFC_LeNet_5"

elif model_name == "Deep_RetiNet":
  model = Deep_RetiNet(depth, rks, in_channels).to(device)
  model_save_name = f"Deep_RetiNet_d{depth}_rks{rks}"

elif model_name == "BNLeNet":
  model = BNDFC_LeNet_5(in_channels).to(device)
  model_save_name = f"BNDFC_LeNet_5"

elif model_name == "INLeNet":
  model = INDFC_LeNet_5(in_channels).to(device)
  model_save_name = f"INDFC_LeNet_5"


optimizer = optimizer(model.parameters(), lr=start_lr)

print(model)

trainer = Trainer(model)


gc.collect()

trainer.train(trainloader = trainloader,
              validloader = testloader,
              optimizer = optimizer,
              loss_fn = loss_fn,
              epochs = frac_epochs,
              plotting = False)

gc.collect()

trainer.train(trainloader = trainloader,
              validloader = testloader,
              optimizer = optimizer,
              loss_fn = loss_fn,
              epochs = frac_epochs,
              plotting = False)

gc.collect()

trainer.train(trainloader = trainloader,
              validloader = testloader,
              optimizer = optimizer,
              loss_fn = loss_fn,
              epochs = frac_epochs,
              plotting = False)

gc.collect()

trainer.train(trainloader = trainloader,
              validloader = testloader,
              optimizer = optimizer,
              loss_fn = loss_fn,
              epochs = frac_epochs,
              plotting = False)
gc.collect()

for g in optimizer.param_groups:
  g["lr"] = 1e-5

trainer.train(trainloader = trainloader,
              validloader = testloader,
              optimizer = optimizer,
              loss_fn = loss_fn,
              epochs = frac_epochs,
              plotting = False)

PATH_TO_SAVE = PATH_TO_ROOT + f"/trained_models/{dataset_name}"
torch.save(model.state_dict(),
            f"{PATH_TO_SAVE}/trained_{model_save_name}_state_dict.pt")
