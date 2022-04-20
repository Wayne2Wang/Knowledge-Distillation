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
        self.drop = torch.nn.Dropout(p=0.3)

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

class MLP_drop_bnorm3(torch.nn.Module):
    """
    # Got best performance using:
        -lr 1e-3
        -hs 512 512 256 256 128 128
        -epoch 40 (it is keep getting better!) (57.4% val acc at 56-th epoch)
    # Beats MLPonly / drop only
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP_drop_bnorm3, self).__init__()
        self.input_size = np.prod(input_size)
        self.hidden_sizes  = hidden_sizes
        self.output_size = output_size

        sizes = [self.input_size] + hidden_sizes + [output_size]
        
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.bnorm = torch.nn.ModuleList([torch.nn.BatchNorm1d(sizes[i+1]) for i in range(len(sizes)-2)])
        self.relu = torch.nn.ReLU()
        self.hooklayer = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(p=0.3)

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
        output = self.hooklayer(output) # hooklayer to match
        output = self.drop(output)
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
            #torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=0) # (128, 1, 1) # we don't need this lol
        )
        self.fc = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.cnn_layers(x) # 3, 28, 28 -> 128, 3, 3
        x = x.mean([2, 3]) # 128,
        #x = torch.flatten(x, start_dim=1) # 128
        x = self.fc(x)
        return x

class CNN_Resblock(torch.nn.Module):
    """
    Residual Block
    """
    def __init__(self, input_dim, intermediate_dim):
        super(CNN_Resblock, self).__init__()
        self.cnn1 = torch.nn.Conv2d(input_dim, intermediate_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(intermediate_dim)
        self.relu1 = torch.nn.ReLU()
        
        self.cnn2 = torch.nn.Conv2d(intermediate_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(input_dim)

        self.reluOut = torch.nn.ReLU()
    
    def forward(self, x):
        out = self.cnn1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.cnn2(out)
        out = self.bn2(out)

        out = self.reluOut(out + x)
        return out

class CNN_Block(torch.nn.Module):
    """
    Normal CNN block
    """
    def __init__(self, input_dim, output_dim):
        super(CNN_Block, self).__init__()
        self.cnn1 = torch.nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_dim)
        self.relu1 = torch.nn.ReLU()
    def forward(self, x):
        out = self.cnn1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        return out

class CNN_JP3(torch.nn.Module):
    # input size for CIFAR10 is 3x32x32
    def __init__(self, input_dim, output_dim):
        super(CNN_JP3, self).__init__()
        self.cnnblock1 = CNN_Block(input_dim, 128)
        #self.cnnblock2 = CNN_Block(64, 128)

        #self.resblock1 = CNN_Resblock(64, 32) # residual blocks. (input_channels -> input_channels, second parameter is intermediate channels which I arbitratily set it as half input channels)
        self.hooklayer = CNN_Resblock(128, 64) # special naming scheme for hook layers (KD Matching layer)

        self.mpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.drop = torch.nn.Dropout(p=0.2) # dropout final output
        self.fc = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.cnnblock1(x) # 128x32x32
        x = self.mpool(x) # 64x16x16
        x = self.hooklayer(x) # 128x16x16
        
        x = x.mean([2,3]) # 128,
        x = self.drop(x) # dropout final layer
        x = self.fc(x)
        return x

class CNN_JP4(torch.nn.Module):
    # input size for CIFAR10 is 3x32x32
    def __init__(self, input_dim, output_dim):
        super(CNN_JP4, self).__init__()
        self.cnnblock1 = CNN_Block(input_dim, 64)
        self.cnnblock2 = CNN_Block(64, 64)
        self.cnnblock3 = CNN_Block(64, 128)
        self.hooklayer = CNN_Block(128, 128)

        self.mpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.drop2d = torch.nn.Dropout2d(p=0.1) # dropout intermediate layer
        self.drop = torch.nn.Dropout(p=0.3) # dropout final output
        self.fc = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.cnnblock1(x) # 64x32x32
        x = self.mpool(x) # 64x16x16
        x = self.drop2d(x)

        x = self.cnnblock2(x) # 64x16x16
        x = self.mpool(x) # 64x8x8
        x = self.drop2d(x)

        x = self.cnnblock3(x) #128x8x8
        x = self.mpool(x) # 128x4x4
        x = self.drop2d(x)

        x = self.hooklayer(x) # 128x4x4
        
        x = x.mean([2,3]) # 128,
        x = self.drop(x) # dropout final layer
        x = self.fc(x)
        return x