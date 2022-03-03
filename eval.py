
import torch
import torchvision
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import ImageNet
from upscaler import ModelUpscaler

"""
A script to evaluate the accuracy of the pretrained resnets (to be extended to other models)
"""


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
    num_batches = 1000
    verbose = True # number of batches to run evaluation, set to inf to run on the whole dataset

    dataset = 'ImageNet64'
    trainset, valset = ImageNet(root='data/{}/'.format(dataset), flat=False)
    model = torchvision.models.resnet50(pretrained=True).to(device)
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