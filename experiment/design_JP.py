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
    """
    # Got best performance using:
        -lr 1e-3
        -hs 512 512 256 256 128 128
        -epoch 40 (it is keep getting better!) (57.4% val acc at 56-th epoch)
    # Beats MLPonly / drop only
    """
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

class MLP_bnorm(torch.nn.Module):
    """
    # Got best performance using:
        -lr 1e-3
        -hs 512 512 256 256 128 128
        -epoch 40 (it is keep getting better!) (57.4% val acc at 56-th epoch)
    # MLP_bnorm only (no dropout)
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP_bnorm, self).__init__()
        self.input_size = np.prod(input_size)
        self.hidden_sizes  = hidden_sizes
        self.output_size = output_size

        sizes = [self.input_size] + hidden_sizes + [output_size]
        
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.bnorm = torch.nn.ModuleList([torch.nn.BatchNorm1d(sizes[i+1]) for i in range(len(sizes)-2)])
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        num = len(self.fcs)
        output = x
        output = torch.flatten(x, start_dim=1)
        for i in range(num-1):
            output = self.fcs[i](output)
            output = self.bnorm[i](output)
            output = self.relu(output)
        output = self.fcs[num-1](output)
        return output

class MLP_drop_bnorm2(torch.nn.Module):
    """
    # Got best performance using:
        -lr 1e-3
        -hs 512 512 256 256 128 128
        -epoch 40 (it is keep getting better!) (57.4% val acc at 56-th epoch)
    # Beats MLPonly / drop only
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP_drop_bnorm2, self).__init__()
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
        for i in range(num-2):
            output = self.fcs[i](output)
            output = self.bnorm[i](output)
            output = self.relu(output)
            output = self.drop(output) # added dropout
        output = self.fcs[num-2](output)
        output = self.bnorm[num-2](output)
        output = self.relu(output) # removed dropout at the last layer for feature matching
        output = self.fcs[num-1](output)
        return output

class CNN_JP1(torch.nn.Module):
    # input size for MNIST is 1x28x28
    def __init__(self, input_dim, output_dim):
        super(CNN_JP1, self).__init__()
        self.cnn_layers = torch.nn.Sequential( # (1, 28, 28)
            torch.nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1), # (64, 28, 28)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (64, 14, 14)

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (128, 14, 14)
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (128, 7, 7)

            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # (256, 7, 7)
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (256, 3, 3)
            
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # (256, 3, 3)
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=0) # (256, 1, 1)
        )
        self.fc = torch.nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.cnn_layers(x) # 3, 28, 28 -> 256, 1, 1
        x = torch.flatten(x, start_dim=1) # 256
        x = self.fc(x)
        return x

class CNN_JP2(torch.nn.Module):
    # input size for MNIST is 1x28x28
    def __init__(self, input_dim, output_dim):
        super(CNN_JP2, self).__init__()
        self.cnn_layers = torch.nn.Sequential( # (1, 28, 28)
            torch.nn.Conv2d(input_dim, 128, kernel_size=3, stride=1, padding=1), # (128, 28, 28)
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (128, 14, 14)

            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # (128, 14, 14)
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (128, 7, 7)

            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # (256, 7, 7)
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (128, 3, 3)
            
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # (128, 3, 3)
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=0) # (128, 1, 1)
        )
        self.fc = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.cnn_layers(x) # 3, 28, 28 -> 128, 1, 1
        x = torch.flatten(x, start_dim=1) # 128
        x = self.fc(x)
        return x