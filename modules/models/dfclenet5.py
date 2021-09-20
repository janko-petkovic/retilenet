# Fully connected version of LeNet 5: the c3 layer does not follow
# the connection matrix from the article but is just a straighforward
# Conv2d with 6in and 16 out. Everybody on the internet uses this but
# i don't know why they call it LeNet5, im probably dumb.
#
# DFC_LeNet_5(in_channels : int)


import torch
import torch.nn as nn


class DFC_LeNet_5(nn.Module):

  def __init__(self, 
               in_channels: int = 1):

    super().__init__()

    self.tanh = nn.Tanh()
    self.c1 = nn.Conv2d(in_channels,6,kernel_size=5, stride=1, padding=0)
    self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

    self.c3 = nn.Conv2d(6,16,kernel_size=5, stride=1)
    self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
    self.c5 = nn.Conv2d(16,120,kernel_size=5,stride=1)

    self.l1 = nn.Linear(120,84)
    self.l2 = nn.Linear(84,10)

    self.tanh = nn.Tanh()
    self.drop = nn.Dropout(0.2)
  


  def forward(self,x):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dim = x.shape[0]
    out = self.c1(x)
    out = self.drop(out)
    out = self.tanh(out)

    out = self.s2(out)
    out = self.c3(out)
    out = self.drop(out)
    out = self.tanh(out)

    out = self.s4(out)
    out = self.c5(out)
    out = self.drop(out)

    temp = torch.zeros(dim,120).to(device)

    for idx in range(dim):
      temp[idx,:] = out[idx,:,0,0]
    
    out = temp
    out = self.tanh(out)
    out = self.l1(out)
    out = self.tanh(out)
    out = self.l2(out)

    return out




# testing section
def test(in_ch):

  from pytorch_model_summary import summary
  
  model = DFC_LeNet_5(in_ch)
  input = torch.zeros(1,in_ch,32,32)

  print("\n","-"*70)
  print(f"Input shape: {input.shape}")
  print(summary(model,
                input,
                show_input=False))


if __name__ == "__main__":

  import sys
  
  in_ch = int(sys.argv[1])
  test(in_ch)
  print("\nTesting successful!\n")
