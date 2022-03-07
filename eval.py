
import torch
import torchvision
import torchmetrics
import torchsummary
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import *
from utils.upscaler import ModelUpscaler
import argparse

"""
A script to evaluate the accuracy of the pretrained resnets (to be extended to other models)
"""

def parse_arg():
    #Parse the command line arguments
    parser = argparse.ArgumentParser(description='Arugments for evaluation')
    #parser.add_argument('--resnet', action='store_true', help='If True, train a resnet(for testing purpose)')
    parser.add_argument('--modelpath', type=str, default='MLP', help='path to the model you are tryng to evaluate')
    parser.add_argument('--augment', action='store_true', help='If True, evaluate on augmented dataset')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to evaluate on')
    parser.add_argument('--root', default='data/CIFAR10', help='Set root of dataset')
    parser.add_argument('--num_batches', type=int, default=1000, help='Max number of batches to evaluate on')
    
    args = parser.parse_args()
    return args


def eval_acc(model, data_loader, device, num_batches=None, verbose=False, mode='validation'):
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
    
    #dataset = 'ImageNet64'
    #trainset, valset = ImageNet(root='data/{}/'.format(dataset), flat=False, evalmode=True)
    if dataset=='ImageNet64':
        trainset, valset = ImageNet(root='D:/Research/Dataset/ImageNet64_Zilin/ImageNet64/', flat=False, evalmode=augment_data)
        if modelpath=='resnet':
            model = torchvision.models.resnet50(pretrained=True).to(device)
        else:
            model = torch.load(modelpath)
    elif dataset=='ImageNet1k':
        trainset, valset = ImageNet(root=root, flat=False, evalmode=augment_data)
        if modelpath=='resnet':
            model = torchvision.models.resnet50(pretrained=True).to(device)
        else:
            model = torch.load(modelpath)
    elif dataset=='CIFAR10':
        trainset, valset = CIFAR(root=root, flat=False, evalmode=augment_data)
        if modelpath=='resnet':
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", 'cifar10_resnet20', pretrained=True)
        else:
            model = torch.load(modelpath)
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
    eval_acc(model, train_loader, device, verbose=verbose, num_batches=num_batches, mode='train')

    # evaluation on valset: Validation acc1 is 0.75390625, validation acc5 is 0.931640625 over 4 batchs
    eval_acc(model, val_loader, device, verbose=verbose, num_batches=num_batches, mode='validation')


if __name__ == '__main__':
    main()
