"""
Loads checkpoint and save it into a model

## not needed. just call train.py, checkpoint, with 0-epoch
"""


from utils.checkpoint import load_checkpoint
from torchsummary import summary
import argparse
import torch
import os

import sys
sys.path.insert(0, "..") # add parent 
from experiment.design_JP import * # call your models here
from models import MLP


def parse_arg():
    # --cp_path log/CIFAR10/MLP_drop_bnorm_56.pt --model MLP_drop_bnorm --hs 512 512 256 256 128 128
    parser = argparse.ArgumentParser(description='Checkpoint to model saver argument parser')
    parser.add_argument('--cp_path', default='data/ImageNet64', help='Path to checkpoint')
    parser.add_argument('--model', default='MLP', help="Model used to train")
    parser.add_argument('--hs', default=[2000, 1000, 100], nargs="*", type=int, help='Hidden unit setting')
    parser.add_argument('--inputsize', default=[3, 32, 32], nargs="*", type=int, help='Input size (default=CIFAR10)')
    parser.add_argument('--outputsize', default=10, nargs="*", type=int, help='Output size (default=CIFAR10)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--reg', default=1e-3, type=float, help="Specify the strength of the regularizer")
    parser.add_argument('--criterion', default='default', help='Criterion (default=non_KD, KD=KD)')
    args = parser.parse_args()
    return args

def checkpoint_loader(model, cp_path, optimizer, criterion):
    # optimizer = adam, etc..
    # loss = criterion = crossentropyloss etc
    epoch, best_loss, train_acc, val_acc = load_checkpoint(model, optimizer, criterion, cp_path, verbose=True)
    cpdir = os.path.dirname(cp_path)
    torch.save(model, os.path.join(cpdir, "{}_{}_model_ld.pt".format(type(model).__name__, epoch)))
    print("Model saved as {}/{}_{}_model_ld.pt".format(cpdir, type(model).__name__, epoch))
    
def main():
    args = parse_arg()
    cp_path = args.cp_path
    modelname = args.model
    hidden_sizes = args.hs
    criterion = args.criterion
    lr = args.lr
    regstr = args.reg
    
    input_size = args.inputsize
    output_size = args.outputsize
    
    modelmethod = globals()[modelname] # call function with 'modelname' name
    model = modelmethod(input_size, hidden_sizes, output_size)
    summary(model, (1,)+tuple(input_size), device='cpu')
    
    if criterion=='default':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=regstr)
        criterion = torch.nn.CrossEntropyLoss()
    elif criterion=='KD':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=regstr)
        criterion = torch.nn.CrossEntropyLoss()
        
    # load and save model
    checkpoint_loader(model, cp_path, optimizer, criterion)
        
        