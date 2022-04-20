
import torch
import torchvision
import torchmetrics
import torchsummary
import pandas as pd
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import *
from utils.upscaler import ModelUpscaler
import argparse
from datetime import datetime

"""
A script to evaluate the accuracy of the pretrained resnets (to be extended to other models)
"""

def parse_arg():
    #Parse the command line arguments
    ## Last arg: --modelpath resnet --dataset CIFAR10 --root data/CIFAR10
    parser = argparse.ArgumentParser(description='Arugments for evaluation')

    parser.add_argument('--modelpath', type=str, default='MLP', help='path to the model you are tryng to evaluate')
    parser.add_argument('--augment', type=float, default=-1, help='Set intensity for augmented dataset evaluation (-1:off)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to evaluate on')
    parser.add_argument('--root', default='data/CIFAR10', help='Set root of dataset')
    parser.add_argument('--num_batches', type=int, default=1e9, help='Max number of batches to evaluate on (Default: 1e9)')
    parser.add_argument('--outfilepath', default='', help='output result to this file if specified')
    # MNIST-C Specific settings
    parser.add_argument('--augtype', type=str, default='translate', help='Data augmentation type for MNIST-C dataset. Will focus mostly on scale and translate')
    
    args = parser.parse_args()
    return args


def eval_acc(model, data_loader, device, num_batches=None, verbose=False, mode='validation', write_file=False, intensity=False):
    """
    Calculate the top1 and top5 accuracy of the model

    Input:
    - model: The model to be evaluated
    - data_loader: The data to use for evaluation
    - device: The device to use
    - num_batches: The number of batches from the data loader to be used
    - mode: train/validation
    - verbose: If True print necessary information
    """

    metric1 = torchmetrics.Accuracy(top_k=1).to(device)
    metric5 = torchmetrics.Accuracy(top_k=5).to(device)
    iterator = tqdm(data_loader, ascii=True, desc='Evaluating {} accuracy: '.format(mode)) if verbose else data_loader
    num_batches = float('inf') if not num_batches else num_batches
    
    model.eval()
    num_batch=0
    for batch in iterator:
        if num_batch>num_batches:
            break
        num_batch += 1
        with torch.no_grad():
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            if type(model).__name__ == 'MLP':
                # flat for MLPs
                preds_onehot = model(x_batch.reshape(x_batch.shape[0],-1))
            else:
                preds_onehot = model(x_batch)
            metric1.update(preds_onehot, y_batch)
            metric5.update(preds_onehot, y_batch)
    acc1 = metric1.compute()
    acc5 = metric5.compute()
    model.train()
    if verbose:
        print('Top1 accuracy is {}, top5 accuracy is {} over {} batches'.format(acc1, acc5, num_batches+1 if not num_batches == float('inf') else 'all'))
    if write_file != '': # if not empty
        with open('{}'.format(write_file), mode='a') as f:
            f.write("Current time: {}\n".format(datetime.now()))
            if intensity:
                f.write("Augmentation intensity: {}\n".format(intensity))
            f.write("[{}] Top1: {}, Top5: {}\n\n".format(mode, acc1, acc5))
        
    return acc1, acc5



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cpu'
     # number of batches to evaluate
    verbose = True # number of batches to run evaluation, set to inf to run on the whole dataset
    
    args = parse_arg()
    modelpath = args.modelpath
    augment_data = args.augment
    dataset = args.dataset
    root = args.root
    num_batches = args.num_batches
    write_file = args.outfilepath
    
    #dataset = 'ImageNet64'
    #trainset, valset = ImageNet(root='data/{}/'.format(dataset), flat=False, evalmode=True)
    if dataset=='ImageNet64':
        trainset, valset = ImageNet(root='D:/Research/Dataset/ImageNet64_Zilin/ImageNet64/', flat=False, evalmode=(True if augment_data >= 0 else False), intensity=augment_data)
        if modelpath=='resnet':
            model = torchvision.models.resnet50(pretrained=True).to(device)
        else:
            model = torch.load(modelpath)
    elif dataset=='ImageNet1k':
        trainset, valset = ImageNet(root=root, flat=False, evalmode=(True if augment_data >= 0 else False), intensity=augment_data)
        if modelpath=='resnet':
            model = torchvision.models.resnet50(pretrained=True).to(device)
        else:
            model = torch.load(modelpath)
    elif dataset=='CIFAR10':
        trainset, valset = CIFAR(root=root, flat=False, evalmode=(True if augment_data >= 0 else False), intensity=augment_data)
        if modelpath=='resnet':
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", 'cifar10_resnet20', pretrained=True)
            model = model.cuda()
        else:
            model = torch.load(modelpath)
    elif dataset=='MNIST':
        trainset, valset =  MNIST(root=root, flat=False, evalmode=(True if augment_data >= 0 else False), intensity=augment_data)
        if modelpath=='CNN_MNIST':
            model = torch.load('assets/CNN_MNIST_10_model.pt')
            model = model.cuda()
        else:
            model = torch.load(modelpath)
    elif dataset == 'MNIST_C':
        trainset, valset = MNIST_C(root=root, verbose=verbose, type=args.augtype)
        if modelpath=='CNN_MNIST':
            model = torch.load('assets/CNN_MNIST_10_model.pt')
            model = model.cuda()
        else:
            model = torch.load(modelpath)
    else:
        raise Exception(dataset+' dataset not supported!')
    #torchsummary.summary(model, (3, 64, 64))
    
    """
    # Load previously trained model if specified
    if not load_model == '':
        prev_epoch, _, _, _ = load_checkpoint(model, optimizer, criterion, load_model, verbose)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=regstr)
    criterion = torch.nn.CrossEntropyLoss()
    model_name = type(model).__name__
    model = model.to(device) # avoid different device error when resuming training
    """
    
    # upscaler
    #model = ModelUpscaler(model, 224)
    train_loader = DataLoader(trainset, batch_size = 128, num_workers = 3, shuffle = False)
    val_loader = DataLoader(valset, batch_size = 128, num_workers = 3, shuffle = False)
    print('Model = {}'.format(type(model).__name__))

    # evaluation on trainset: Train acc1 is 0.966796875, train acc5 is 0.99609375 over 4 batchs
    train_acc1, train_acc5 = eval_acc(model, train_loader, device, verbose=verbose, num_batches=num_batches, mode='train', write_file=write_file, intensity=augment_data)

    # evaluation on valset: Validation acc1 is 0.75390625, validation acc5 is 0.931640625 over 4 batchs
    validation_acc1, validation_acc5 =  eval_acc(model, val_loader, device, verbose=verbose, num_batches=num_batches, mode='validation', write_file=write_file, intensity=augment_data)

    

    if (write_file != ''):
        try: 
            results_df = pd.read_csv(write_file)
            results_df = results_df.dropna(axis=0)
            results_df = results_df.dropna(axis=1)
            results_df = results_df.drop(columns=['Unnamed: 0']) # what is this row lol
        except:
            results_df = pd.DataFrame()
        data = [[type(model).__name__, os.path.basename(modelpath), dataset, train_acc1.item(),\
            validation_acc1.item(), augment_data]]
        d = pd.DataFrame(data, columns = ['Student Model', 'Model Path', 'Dataset', 'Train Acc', 'Validation Acc', 'Augmentation Intensity'])
        results_df = pd.concat((results_df, d), axis=0, ignore_index = True)

        results_df.to_csv(write_file)


if __name__ == '__main__':
    main()
