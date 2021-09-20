# A custom conv2d mimicking the retinic color pathway segregation:
# while L and M pipelines interact with each other (in an antagonistic
# center-surround fashion in the human visual system), the S route
# is kept separate, following the dedicated koniocellular pathway.

# The module is basically a wrapper of a 


#   nn.Conv2D(in_channels = 3,
#             out_channels = 3,
#             kernel_size = xxxxxx,    <-   you can choose how big your
#                                           receptive field is
#             stride = 1,
#             padding = int((kernel_size-1)/2))


# where the first two outputs don't see the third input and the third
# output doesn't see the first two inputs.


# CAVEAT: to train this module it is necessary to use the custom Trainer
# rememberting the retitrain = True flag. If not the segregations will not
# be considered during parameter update.


import torch
import torch.nn as nn



class Conv2dRetina(nn.Module):
  def __init__(self, kernel_size):
    super().__init__()
    
    # automatic even padding
    if (kernel_size+1) % 2:
      raise Exception("Kernel size must be even")
    
    else:
      padding = int((kernel_size-1) / 2)
    
    self.conv = nn.Conv2d(in_channels = 3,
                          out_channels = 3,
                          kernel_size = kernel_size,
                          stride= 1,
                          padding = padding)
    
    #self._split_color_pathways(kernel_size)
    
  # different cones have different pathways: L and M interact
  # while S cones are segregated to the koniocellular pathway
  def _split_color_pathways(self, kernel_size):
    for i, cube in enumerate(self.conv.weight):
      if i != 2:
        cube.data[2] = torch.zeros(1,kernel_size,kernel_size)
      else:
        cube.data[:2] = torch.zeros(2,kernel_size,kernel_size)
        

  def forward(self, x):
    return self.conv(x)



# testing section
def test(channels):

  from pytorch_model_summary import summary
  
  model = Conv2dRetina(channels)
  input = torch.zeros(1,channels,28,28)
  print("\n","-"*70)
  print(f"Input shape: {input.shape}")
  print(summary(model,
                torch.zeros(1,3,28,28),
                show_input=False))

if __name__ == "__main__":

  import sys
  
  channels = int(sys.argv[1])
  test(channels)
  print("\nTesting successful!\n")
