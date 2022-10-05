import torch, random
from torchvision import datasets, transforms

import matplotlib.pyplot as plt




# Custom transforms for which we are testing
class ShiftMeanTransform:
    def __init__(self, mu_min: float, mu_max: float):
        self.mu_min = mu_min
        self.mu_max = mu_max

    def __call__(self, x):
        mu = random.uniform(self.mu_min, self.mu_max)
        return x+mu

class ScaleVarTransform:
    def __init__(self, s_min, s_max):
        self.s_min = s_min
        self.s_max = s_max

    def __call__(self, x):
        s = random.uniform(self.s_min, self.s_max)
        return (x-x.mean())/s + x.mean()



transform = transforms.Compose([transforms.ToTensor(),
              ShiftMeanTransform(-2.,2.),
              ScaleVarTransform(-0.1, 4.)])


mnist = datasets.MNIST(
            './datasets',
            download = True,
            train = True, 
            transform = transform
            )

fmnist = datasets.FashionMNIST(
            './datasets',
            download = True,
            train = True, 
            transform = transform
            )

svhn = datasets.SVHN(
            './datasets/SVHN',
            download = True,
            split = "train",
            transform = transform
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

fig, axs = plt.subplots(1,2,figsize=(10,4))
fig.tight_layout(pad=2)

for name, _np in zip(['MNIST', 'FashionMNIST', 'SVHN'], [mnist_np, fmnist_np, svhn_np]):
    means = _np.mean(axis=2).squeeze()
    stds = _np.std(axis=2).squeeze()
    axs[0].hist(means, bins=10, alpha=0.5, label=name)
    axs[1].set_xscale('log')
    axs[1].hist(stds, bins=10, alpha=0.5, label=name)

axs[0].set_title('Mean pixel value')
axs[1].set_title('Pixel value distribution')

for ax in axs:
    ax.set_xlabel('value [U.A.]')
    ax.set_ylabel('counts')
    plt.legend()

plt.show()
