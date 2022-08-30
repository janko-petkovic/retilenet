import torch
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

mnist = datasets.MNIST(
            './datasets',
            download = True,
            train = True, 
            transform = transforms.ToTensor()
            )

fmnist = datasets.FashionMNIST(
            './datasets',
            download = True,
            train = True, 
            transform = transforms.ToTensor()
            )

svhn = datasets.SVHN(
            './datasets/SVHN',
            download = True,
            split = "train",
            transform = transforms.ToTensor()
            )

mnist_size = mnist.__len__()
fmnist_size = fmnist.__len__()
svhn_size = svhn.__len__()

mnist_l = torch.utils.data.DataLoader(
            mnist,
            batch_size = mnist_size)
fmnist_l = torch.utils.data.DataLoader(
            fmnist,
            batch_size = fmnist_size)
svhn_l = torch.utils.data.DataLoader(
            svhn,
            batch_size = svhn_size)

mnist_np = 0
fmnist_np = 0
svhn_np = 0

for batch, _ in mnist_l:
    mnist_np = batch.reshape(len(batch), 1,-1).cpu().detach().numpy()
for batch, _ in fmnist_l:
    fmnist_np = batch.reshape(len(batch), 1,-1).cpu().detach().numpy()
for batch, _ in svhn_l:
    svhn_np = batch.reshape(len(batch), 1,-1).cpu().detach().numpy()

fig, axs = plt.subplots(1,2,figsize=(5,2))
fig.tight_layout(pad=2)

for name, _np in zip(['MNIST', 'FashionMNIST', 'SVHN'], [mnist_np, fmnist_np, svhn_np]):
    means = _np.mean(axis=2).squeeze()
    stds = _np.std(axis=2).squeeze()
    axs[0].hist(means, bins=10, alpha=0.5, label=name)
    axs[1].hist(stds, bins=10, alpha=0.5, label=name)

axs[0].set_title('Mean pixel value')
axs[1].set_title('Pixel value distribution')

for ax in axs:
    ax.set_xlabel('value [U.A.]')
    ax.set_ylabel('counts')
    plt.legend()

plt.show()
