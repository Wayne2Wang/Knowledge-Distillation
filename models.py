import time
import torch
import torchmetrics

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.checkpoint import save_checkpoint
from eval import eval_acc
import numpy as np

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.input_size = np.prod(input_size)
        self.hidden_sizes  = hidden_sizes
        self.output_size = output_size
        
        sizes = [self.input_size] + hidden_sizes + [output_size]
        
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        num = len(self.fcs)
        output = x
        output = torch.flatten(x, start_dim=1)
        for i in range(num-1):
            output = self.fcs[i](output)
            output = self.relu(output)
        output = self.fcs[num-1](output)
        return output


def fit_model(
    model,
    data, 
    optimizer,
    criterion,
    device='cpu', 
    prev_epoch=0,
    epochs=10,  
    batch_size=64, 
    save_every=5,
    verbose=False):

    model_name = type(model).__name__

    # Read data
    trainset, valset, dataset = data
    train_size = len(trainset)
    train_loader = DataLoader(trainset, batch_size = batch_size, num_workers = 4, shuffle = False)
    val_loader = DataLoader(valset, batch_size = batch_size, num_workers = 4, shuffle = False)
  
    # Start training
    writer = SummaryWriter(log_dir='log\\{}\\{}'.format(dataset, model_name))
    start_time = time.time()
    best_loss = float('inf')
    best_train_acc1 = 0
    best_train_acc5 = 0
    best_val_acc1 = 0
    best_val_acc5 = 0
    real_epoch = prev_epoch
    model = model.to(device)
    metric1 = torchmetrics.Accuracy(top_k=1).to(device)
    metric5 = torchmetrics.Accuracy(top_k=5).to(device)

    for epoch in range(epochs):

        total_loss = 0
        real_epoch = prev_epoch+epoch+1
        model.train()
        for batch in tqdm(train_loader, ascii=True, desc='Epoch {}/{}'.format(real_epoch, prev_epoch+epochs)):
            # Prepare minibatch
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)

            # Clear gradient
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x_batch)

            # Compute Loss
            loss = criterion(y_pred, y_batch)

            # Computer train acc
            metric1.update(y_pred, y_batch)
            metric5.update(y_pred, y_batch)

            # Record loss
            loss_value = loss.item()
            total_loss += loss_value

            # Backward pass
            loss.backward()
            optimizer.step()

        # Evaluate this epoch
        total_loss /= train_size
        train_acc1 = metric1.compute()
        train_acc5 = metric5.compute()
        metric1.reset()
        metric5.reset()
        val_acc1, val_acc5 = eval_acc(model, val_loader, device) 

        best_train_acc1 = train_acc1 if train_acc1 > best_train_acc1 else best_train_acc1
        best_train_acc5 = train_acc5 if train_acc5 > best_train_acc5 else best_train_acc5
        best_val_acc1 = val_acc1 if val_acc1 > best_val_acc1 else best_val_acc1
        best_val_acc5 = val_acc5 if val_acc5 > best_val_acc5 else best_val_acc5
        best_loss = total_loss if total_loss < best_loss else best_loss

        # log training stats
        writer.add_scalar('Loss/train', total_loss, real_epoch)
        writer.add_scalar('Accuracy1/train', train_acc1, real_epoch)
        writer.add_scalar('Accuracy5/train', train_acc5, real_epoch)
        writer.add_scalar('Accuracy1/val', val_acc1, real_epoch)
        writer.add_scalar('Accuracy5/val', val_acc5, real_epoch)
        
        if verbose:
            print('train loss {:5f}, train_acc1 {:5f}, train_acc5 {:5f}, val_acc1 {:5f}, val_acc5 {:5f}, time {:2f}s'.format(total_loss, train_acc1, train_acc5, val_acc1, val_acc5, time.time()-start_time))

        train_acc = train_acc1, train_acc5
        val_acc = val_acc1, val_acc5
        if (epoch+1)%save_every == 0:
            save_checkpoint('{}\\{}_{}.pt'.format(dataset, model_name, real_epoch),model, real_epoch, optimizer, criterion, total_loss, train_acc, val_acc, verbose=verbose)


    if not epochs%save_every == 0:
        save_checkpoint('{}\\{}_{}.pt'.format(dataset, model_name, real_epoch),model, real_epoch, optimizer, criterion, total_loss, train_acc, val_acc, verbose=verbose)

    total_time = time.time() - start_time
    best_train_acc = best_train_acc1, best_train_acc5
    best_val_acc = best_val_acc1, best_val_acc5

    return best_loss, best_train_acc, best_val_acc, total_time
            

    
