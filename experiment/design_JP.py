# Model design for Jeongsoo Park
# Github desktop test
import torch

class MLP_JS_V1(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, reg_str):
        super(MLP_JS_V1, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size = output_size
        self.reg_str = reg_str

        sizes = [input_size] + hidden_sizes + [output_size]
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        num = len(self.fcs)
        output = x
        for i in range(num-1):
            output = self.fcs[i](output)
            output = self.relu(output)
        output = self.fcs[num-1](output)
        return output