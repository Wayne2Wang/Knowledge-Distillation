import argparse

import torch
import torchvision
from torchsummary import summary

from datasets import *
from models import *
from experiment.design_JP import * # call your models here
from utils.checkpoint import load_checkpoint

def parse_arg():
    """
    Parse the command line arguments
    """
    # Last run arg: --model MLP_drop_bnorm --load_model log/CIFAR10/MLP_drop_bnorm_40.pt --lr 1e-3 --bs 256 --hs 512 512 256 256 128 128 --dataset CIFAR10 --verbose --epochs 20 --root "D:/Research/Dataset/CIFAR10"
    parser = argparse.ArgumentParser(description='Arugments for fitting the feedforward model')
    
    ### most frequently used arguments
    parser.add_argument('--model', type=str, default='MLP', help='Name of the model you\'re going to train')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset used for training')
    parser.add_argument('--root', default='data/CIFAR10', help='Set root of dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    
    ### training settings
    parser.add_argument('--verbose', action='store_true', help='If True, print training progress')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument("--hs",  nargs="*",  type=int, default=[2000, 1000, 100], help='Hidden units')
    parser.add_argument('--reg', default=1e-3, type=float, help="Specify the strength of the regularizer")
    # MNIST-C Specific settings
    parser.add_argument('--augtype', type=str, default='translate', help='Data augmentation type for MNIST-C dataset. Will focus mostly on scale and translate')
    
    ### model checkpoint
    parser.add_argument('--load_model', type=str, default='', help='Resume training from load_model if not empty')
    parser.add_argument('--save_every', type=int, default=1, help='Save every x epochs')

    args = parser.parse_args()
    return args


def main():
    # Read the arguments
    args = parse_arg()
    load_model = args.load_model
    verbose = True #args.verbose
    dataset = args.dataset
    save_every = args.save_every
    
    modelname = args.model
    modelmethod = globals()[modelname] # call function with 'modelname' name
    
    root = args.root
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cpu'

    # Hyperparameters
    epochs = args.epochs
    batch_size = args.bs
    lr = args.lr
    hidden_sizes = args.hs
    regstr = args.reg

    # Read data
    if dataset == 'ImageNet1k':    
        trainset, valset = ImageNet(root=root,flat=(False if modelname=='resnet' else True),verbose=verbose)
        output_size = 1000 # number of distinct labels
        input_size = trainset[0][0].shape[0] # input dimensions
    elif dataset == 'ImageNet64':    
        trainset, valset = ImageNet(root=root,flat=(False if modelname=='resnet' else True),verbose=verbose)
        output_size = 1000 # number of distinct labels
        input_size = trainset[0][0].shape[0] # input dimensions
    elif dataset == 'CIFAR10':
        trainset, valset = CIFAR(root=root, flat=False, verbose=verbose)
        output_size = 10
        input_size = trainset[0][0].shape
    elif dataset == 'MNIST':
        trainset, valset = MNIST(root=root, flat=False, verbose=verbose)
        output_size = 10
        input_size = trainset[0][0].shape
    elif dataset == 'MNIST_C':
        trainset, valset = MNIST_C(root=root, verbose=verbose, type=args.augtype)
        output_size = 10
        input_size = trainset[0][0].shape
    else:
        raise Exception(dataset+' dataset not supported!')
    data = trainset, valset, dataset
    
    
    # Model initialization
    if modelname=='resnet':
        model = torchvision.models.resnet50(pretrained=True)
        ## freeze all layers
        numchild=1
        for child in model.children():
            if numchild < 10:
                for param in child.parameters():
                    param.requires_grad = False
            numchild += 1
        ### Unfreeze fully connected layer
        #model.fc.requires_grad = True
    elif modelname=='MLP':
        model = modelmethod(input_size, hidden_sizes, output_size)
        summary(model, (1,)+tuple(input_size), device='cpu')
    elif modelname == 'CNN_MNIST':
        model = modelmethod(input_size[0], output_size)
        summary(model, tuple(input_size), device='cpu')
    else: ### currently only works for MLP
        model = modelmethod(input_size, hidden_sizes, output_size)
        summary(model, (1,)+tuple(input_size), device='cpu')
        
    model_name = type(model).__name__
    model = model.to(device) # avoid different device error when resuming training
    prev_epoch = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=regstr)
    criterion = torch.nn.CrossEntropyLoss()
    
    """
    ### unused code
    if args.reg == 'l1' and not resnet:
        l1_reg = 0
        for fclayer in model.fcs:
            l1_reg += torch.norm(fclayer.weight, 1)
        #criterion += l1_reg
    ############
    """

    # Load previously trained model if specified
    if not load_model == '':
        prev_epoch, _, _, _ = load_checkpoint(model, optimizer, criterion, load_model, verbose)

    
    # Training
    if verbose:
        print('\nStart training {}: epoch={}, prev_epoch={}, batch_size={}, lr={}, save_every={} device={}'\
                                    .format(model_name, epochs, prev_epoch, batch_size, lr, save_every, device))

    best_loss, train_acc, val_acc, total_time = fit_model(model,
                                                          data, 
                                                          optimizer,
                                                          criterion,
                                                          device=device,
                                                          prev_epoch=prev_epoch,
                                                          epochs=epochs,  
                                                          batch_size=batch_size, 
                                                          save_every=save_every,
                                                          verbose=verbose)

    if verbose:
        print('\nTraining finished! \nTime: {:2f}, (best) train loss: {:5f}, train acc1: {:5f}, train acc5: {:5f}, val acc1: {:5f}, val acc5: {:5f}' \
                                                        .format(total_time, best_loss, train_acc[0], train_acc[1], val_acc[0], val_acc[1]))

if __name__ == '__main__':
    main()