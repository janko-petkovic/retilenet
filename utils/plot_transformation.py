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

import os
from re import L
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# PATH, custom imports, device definition

PATH_TO_ROOT = ".."
PATH_TO_DATASETS = os.path.join(
    PATH_TO_ROOT, "datasets"
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device}')


testset = datasets.FashionMNIST(
    PATH_TO_DATASETS,
    download = True,
    train = False,
    transform = transforms.ToTensor()
)

it = iter(testset)
for _ in range(1):
    example = next(it)[0]
example = next(it)[0].squeeze()

mus = [-1, 0, 1]
sigmas = [0.1, 1, 10]

print(example.max(), example.min())

def shift_mu(example, mu):
    return example - mu

def scale_sigma(example, sigma):
    return (example - example.mean())/sigma + example.mean()


fig, axs = plt.subplots(2,3,figsize=(6,4), dpi=300)

for mu, ax in zip(mus, axs[0]):
    ax.imshow(shift_mu(example, mu), cmap='gray', vmin=-1, vmax=2)
    ax.set_title(r'$\mu =$' + f'{mu}')

for sigma, ax in zip(sigmas, axs[1]):
    ax.imshow(scale_sigma(example, sigma), cmap='gray', vmin=0, vmax=1)
    ax.set_title(r'$\sigma =$' + f'{sigma}')
    # ax.hist(scale_sigma(example, sigma).flatten())

for ax in axs.flatten():
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


plt.show()
