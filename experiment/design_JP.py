# Model design for Jeongsoo Park
# Github desktop test
import torch
import numpy as np

class MLP_drop(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP_drop, self).__init__()
        self.input_size = np.prod(input_size)
        self.hidden_sizes  = hidden_sizes
        self.output_size = output_size

        sizes = [self.input_size] + hidden_sizes + [output_size]
        
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        num = len(self.fcs)
        output = x
        output = torch.flatten(x, start_dim=1)
        for i in range(num-1):
            output = self.fcs[i](output)
            output = self.relu(output)
            output = self.drop(output) # added dropout
        output = self.fcs[num-1](output)
        return output
    
class MLP_drop_bnorm(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP_drop_bnorm, self).__init__()
        self.input_size = np.prod(input_size)
        self.hidden_sizes  = hidden_sizes
        self.output_size = output_size

        sizes = [self.input_size] + hidden_sizes + [output_size]
        
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.bnorm = torch.nn.ModuleList([torch.nn.BatchNorm1d(sizes[i+1]) for i in range(len(sizes)-2)])
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        num = len(self.fcs)
        output = x
        output = torch.flatten(x, start_dim=1)
        for i in range(num-1):
            output = self.fcs[i](output)
            output = self.bnorm[i](output)
            output = self.relu(output)
            output = self.drop(output) # added dropout
        output = self.fcs[num-1](output)
        return output