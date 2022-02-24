import time
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.checkpoint import save_checkpoint

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
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
    val_loader = DataLoader(valset, batch_size = batch_size, num_workers = 4, shuffle = True)
  
    # Start training
    writer = SummaryWriter(log_dir='log\\{}\\{}'.format(dataset, model_name))
    start_time = time.time()
    best_loss = float('inf')
    best_train_acc = 0
    best_val_acc = 0
    real_epoch = prev_epoch
    model = model.to(device)

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

            # Record loss
            loss_value = loss.item()
            total_loss += loss_value

            # Backward pass
            loss.backward()
            optimizer.step()

        # Evaluate this epoch
        total_loss /= train_size
        train_acc = 1#utils.eval_model_acc(model,train_loader,num_batch=10, device=device)
        val_acc = 1#utils.eval_model_acc(model,val_loader,num_batch=10, device=device)

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
            print('train loss {:5f}, train_acc {:5f}, val_acc {:5f}, time {:2f}s'.format(total_loss, train_acc, val_acc, time.time()-start_time))

        if (epoch+1)%save_every == 0:
            save_checkpoint('{}\\{}_{}.pt'.format(dataset, model_name, real_epoch),model, real_epoch, optimizer, criterion, total_loss, train_acc, val_acc, verbose=verbose)


    if not epochs%save_every == 0:
        save_checkpoint('{}\\{}_{}.pt'.format(dataset, model_name, real_epoch),model, real_epoch, optimizer, criterion, total_loss, train_acc, val_acc, verbose=verbose)
    total_time = time.time() - start_time

    return best_loss, best_train_acc, best_val_acc, total_time
            

    
# TODO improve efficiency
def eval_model_acc(model, data_loader, num_batch=10, device='cpu'):
    model = model.to(device)
    model.eval()
    correct = 0
    for _ in range(num_batch):
        batch = next(iter(data_loader))
        img = batch[0].to(device)
        label = batch[1].to(device)
        pred = model(img)
        correct += torch.sum((label == torch.argmax(pred, dim=1)).float())
    model.train()
    return correct/(num_batch*img.shape[0])