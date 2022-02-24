
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data import ImageNet1k

"""
A script to evaluate the accuracy of the pretrained resnets
"""


def onehot_accuracy(preds_onehot, y_batch):
    preds = torch.argmax(preds_onehot, dim=1)
    return torch.mean((preds==y_batch).float())

def main():
    batches = 3 # number of batches to run evaluation, set to inf to run on the whole dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cpu'


    trainset, valset = ImageNet1k(flat=False)
    model = torchvision.models.resnet50(pretrained=True).to(device)
    train_loader = DataLoader(trainset, batch_size = 128, num_workers = 3, shuffle = False)
    val_loader = DataLoader(valset, batch_size = 128, num_workers = 3, shuffle = False)

    # evaluation on trainset: Train acc is 0.9661458730697632 over 3 batchs
    num_batch = 0
    train_acc = 0
    model.eval()
    for batch in tqdm(train_loader, ascii=True, desc='Evaluating train accuracy: '):
        if num_batch>batches:
            break
        num_batch += 1
        with torch.no_grad():
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            preds_onehot = model(x_batch)
            train_acc += onehot_accuracy(preds_onehot, y_batch)
    print('Train acc is {} over {} batchs'.format(train_acc/num_batch, num_batch))

    # evaluation on valset: Validation acc is 0.7369791865348816 over 3 batches
    num_batch = 0
    val_acc = 0
    model.eval()
    for batch in tqdm(val_loader, ascii=True, desc='Evaluating validation accuracy: '):
        if num_batch>batches:
            break
        num_batch += 1
        with torch.no_grad():
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            preds_onehot = model(x_batch)
            val_acc += onehot_accuracy(preds_onehot, y_batch)
    print('Validation acc is {} over {} batches'.format(val_acc/num_batch, num_batch))


if __name__ == '__main__':
    main()