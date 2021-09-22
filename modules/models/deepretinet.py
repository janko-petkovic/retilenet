# After discovering that small bistratified cells are peripherally inhibited
# by yellow input, we drop the segregation hypothesis and dive into further
# exploring a very rudimental modelization of the pre-cortical visual pathway.
# Before a DFC_LeNet_5 (a V1 surrogate) we stack a pile of conv2d layers 
# (the abusively called .retina module of this class) which do not modify
# the number of channels/features of the image (in_channels = out_channels)
#
# Deep_RetiNet(depth : int,           ->    number of initial conv2d
#              kernel_size : int,     ->    kernel size of the the stacked convs
#              in_channels : int)     ->    1 or 3 (BW vs colored img)
#
# For our purposes, we will use depth = 3, as we consider a visual pathway made
# up of 3 cellular types:
#
# bipolar cell -> retinal ganglion (optic nerve) ->  LGN output (optic radiation)
#
# This is of course an EXTREMELY simplified model but surprisingly enough, on
# simple datasets, a center-surround receptive field is evolved with a stabilizing
# effect on the downstream hidden outputs.

import torch
import torch.nn as nn

from .dfclenet5 import DFC_LeNet_5

class Deep_RetiNet(nn.Module):
    def __init__(self, depth, kernel_size, in_channels):
        super().__init__()

        self.retina = self._build_retina(depth, kernel_size, in_channels)
        self.lenet = DFC_LeNet_5(in_channels)

    def forward(self, x):
        out = self.retina(x)
        out = self.lenet(out)

        return out


    def _build_retina(self, depth, kernel_size, in_channels):
        layers = []

        padding = int((kernel_size-1)/2)

        for _ in range(depth):
            layers += [
                nn.Conv2d(in_channels = in_channels,
                          out_channels = in_channels,
                          kernel_size = kernel_size,
                          stride = 1,
                          padding = padding),
                nn.Dropout(0.2),
                nn.Tanh()
            ]

        return nn.Sequential(*layers)



# testing section
def test():

  from pytorch_model_summary import summary
  
  model = Deep_RetiNet(10,5,3)
  input = torch.zeros(1,3,32,32)

  print("\n","-"*70)
  print(f"Input shape: {input.shape}")
  print(summary(model,
                input,
                show_input=False))


if __name__ == "__main__":

  test()
  print("\nTesting successful!\n")
