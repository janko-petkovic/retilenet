# RetiNet derived from the DFC Lenet 5. Nothing special really, it just 
# adds an additional conv2d in front of everything to simulate the retinic
# action. We are not using the custom conv2dretina because is seems that in
# the actual human eye there is not color pathway segregation (2019 research
# on small bistratified retinic ganglions).
# 
# DFC_RetiNet(retinic_kernel_size : int,
#             in_channels : int)
#
# NB: in_channels = out_channels for the retinic layer    

import torch
import torch.nn as nn
from .dfclenet5 import DFC_LeNet_5


class DFC_RetiNet(nn.Module):

  def __init__(self,
               retinic_kernel_size: int = 7, 
               in_channels: int = 1):

        super().__init__()

        # kernel size has to be even to preserve input dimension
        if (retinic_kernel_size-1) % 2:
            raise Exception("Kernel size has to be even!")
        else:
            padding = int( (retinic_kernel_size-1)/2 )

        self.retina = nn.Conv2d(
                        in_channels = in_channels,
                        out_channels = in_channels,
                        kernel_size = retinic_kernel_size,
                        stride = 1,
                        padding = padding
                      )      

        self.lenet = DFC_LeNet_5(in_channels)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.2)


  def forward(self, x) -> torch.tensor:
    out = self.retina(x)
    out = self.drop(out)
    out = self.tanh(out)
    out = self.lenet(out)

    return out




# testing section
def test(ks, in_ch):

  from pytorch_model_summary import summary
  
  model = DFC_RetiNet(ks, in_ch)
  input = torch.zeros(1,in_ch,32,32)

  print("\n","-"*70)
  print(f"Input shape: {input.shape}")
  print(summary(model,
                input,
                show_input=False))


if __name__ == "__main__":

  import sys
  
  ks = int(sys.argv[1])
  in_ch = int(sys.argv[2])
  test(ks, in_ch)
  print("\nTesting successful!\n")
