import torch

class feed_forward(torch.nn.Module):
  def __init__(self, input_size, hidden_sizes, output_size):
    super(feed_forward, self).__init__()
    self.input_size = input_size
    self.hidden_sizes  = hidden_sizes
    self.output_size = output_size

    sizes = [input_size] + hidden_sizes + [output_size]
    self.fcs = torch.nn.ModuleList([torch.nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
    self.relu = torch.nn.ReLU()
    #self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    num = len(self.fcs)
    output = x
    for i in range(num-1):
        output = self.fcs[i](output)
        output = self.relu(output)
    output = self.fcs[num-1](output)
    return output