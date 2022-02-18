import torch
import utils
import math
import time
from torch.utils.tensorboard import SummaryWriter

class feed_forward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(feed_forward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size = output_size

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

def fit_ff(
    model,
    data, 
    optimizer,
    criterion,
    prev_epoch=0,
    epochs=100,  
    batch_size=256, 
    save_every=1,
    verbose=False):

    # Read data
    X_train, y_train, X_val, y_val, dataset = data
    train_size = X_train.shape[0]

  
    # Start training
    writer = SummaryWriter(log_dir='log\\{}'.format(dataset))
    start_time = time.time()
    best_loss = float('inf')
    best_train_acc = 0
    best_val_acc = 0
    for epoch in range(epochs):

        # Shuffle the data for each epoch
        shuffler = torch.randperm(train_size)
        X_train_shuffled = X_train[shuffler]
        y_train_shuffled = y_train[shuffler]

        total_loss = 0
        model.train()
        for i in range(math.floor(train_size/batch_size)):

            # Prepare mini-batch
            start = i*batch_size
            if (i+1)*batch_size > train_size:
                end = train_size
            else:
                end = (i+1)*batch_size
                x_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

            # Clear gradient
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x_batch)

            # Compute Loss
            loss = criterion(y_pred, y_batch)

            # Record loss
            loss_value = loss.item()
            total_loss += loss_value

            # Backward pass
            loss.backward()
            optimizer.step()

        # Evaluate this epoch
        real_epoch = prev_epoch+epoch+1
        total_loss /= train_size
        train_acc = utils.eval_model_acc(model,X_train,y_train,ratio=0.1)
        val_acc = utils.eval_model_acc(model,X_val,y_val,ratio=0.1)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if total_loss < best_loss:
            best_loss = total_loss

        # log training stats
        writer.add_scalar('Loss/train', total_loss, real_epoch)
        writer.add_scalar('Accuracy/train', train_acc, real_epoch)
        writer.add_scalar('Accuracy/val', train_acc, real_epoch)
        
        if verbose:
            print('Epoch {}/{}: train loss {:5f}, train_acc {:5f}, val_acc {:5f}, time {:2f}s'.format(real_epoch, prev_epoch+epochs, total_loss, train_acc, val_acc, time.time()-start_time))

        if (epoch+1)%save_every == 0:
            utils.save_checkpoint('{}\\ff_model_{}.pt'.format(dataset, real_epoch),model, real_epoch, optimizer, criterion, total_loss, train_acc, val_acc, verbose=verbose)

    utils.save_checkpoint('{}\\ff_model_{}.pt'.format(dataset, real_epoch),model, real_epoch, optimizer, criterion, total_loss, train_acc, val_acc, verbose=verbose)
    total_time = time.time() - start_time

    return best_loss, best_train_acc, best_val_acc, total_time
            

    
