# CLASS : `Piper(net)`

# A wrapper meant to show the hidden outputs. It has the following 
# public methods:
#   - `define_hooks(tuple)`
#   - `remove_hooks()`
#   - `show_hidden_outputs(n_rows, n_cols, figsize)`

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Piper(nn.Module):
  def __init__(self, net):
    super().__init__()
    self.net = net
    self.handles = []
    self.hidden_outputs = {}

  def _hook_fn(self, module, input, output):
    self.hidden_outputs[module] = output.cpu().detach().numpy()
    

  def _add_hook(self, idx):

    for i, named_child in enumerate(self.net.named_children()):
      name, child = named_child

      if (i == idx):
        print(f'Attaching hook to: {name}')

        handle = child.register_forward_hook(self._hook_fn)

        self.handles.append(handle)


  def remove_hooks(self):
    self.handles = [handle.remove() for handle in self.handles]
    self.handles = []


  def define_hooks(self, indexes):
    self.outputs = []
    self.remove_hooks()
    for idx in indexes:
      self._add_hook(idx)


  def get_hidden_outputs(self):
    hiddens = []
    for key in self.hidden_outputs.keys():
      for hidden_out in self.hidden_outputs[key][0]:
        hiddens.append(hidden_out)

      return hiddens


  def show_hidden_outputs(self, n_rows, n_cols, figsize=(20,5)):
     
    fig = plt.figure(figsize=figsize)
    
    for row, key in enumerate(self.hidden_outputs.keys()):
      for col, filter in enumerate(self.hidden_outputs[key][0]):
        fig.add_subplot(n_rows, n_cols, col+1+row*6)
        filter = filter.squeeze()
        plt.imshow(filter, cmap='gray')
        plt.colorbar()

    plt.show()
    