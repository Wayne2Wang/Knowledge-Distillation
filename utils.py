import torch


def save_checkpoint(path_to_model, model, epoch, optimizer, criterion, loss_value, train_acc, val_acc, verbose=True):
    """
    Save the current training status as a checkpoint

    Input:
    - path_to_model: The location to store the checkpoint
    - model: The model to be checkpointed
    - epoch: Number of epochs this model has been trained
    - optimizer: Optimizer used to optimize the mode
    - criterion: Loss function used to evaluate the model
    - loss_value: Loss value from the last epoch
    - train_acc: Training accuracy from the last epoch
    - val_acc: Validation accuracy from the last epoch
    - verbose: If True print necessary information
    """
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion': criterion,
                'loss_value': loss_value,
                'train_acc' : train_acc,
                'val_acc' : val_acc
                }, path_to_model)
    if verbose:
        print('Saved the model to {}'.format(path_to_model))

def load_checkpoint(model, optimizer, criterion, path_to_model, verbose=True):
    """
    Load the checkpoint from the specified location

    Input:
    - model: The model to be checkpointed
    - optimizer: Optimizer used to optimize the mode
    - criterion: Loss function used to evaluate the model
    - path_to_model: The location to find the checkpoint
    - verbose: If True print necessary information

    Returns:
    - epoch: int storing the number of epoch this model had been trained
    - best_loss: floating number storing the last loss value
    - train_acc: floating number storing the last training accuracy
    - val_acc: floating number storing the last validation accuracy
    """
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #criterion = checkpoint['criterion']
    best_loss = checkpoint['best_loss']
    train_acc = checkpoint['train_acc']
    val_acc = checkpoint['val_acc']
    if verbose:
        print('Loaded checkpoint from {}'.format(path_to_model))
        print('Previous loss {:5f}, train_acc {:5f}, val_acc {:5f}'.format(best_loss, train_acc, val_acc))
    return epoch, best_loss, train_acc, val_acc

def eval_model_acc(model, X, y_true, ratio=0.1):
    """
    Evalute the model on a fraction of the given data

    Input:
    - model: the model to be evaluated
    - X: a tensor storing a mini-batch of data samples
    - y_true: a one-dimensional tensor storing the correct labels
    - ratio: ratio of data samples used to do evaluation

    Returns:
    - result: floating number representing the calculated accuracy
    """
    model.eval()
    size = X.shape[0]
    idx = torch.randperm(size)[:int(size*ratio)]
    X_eval, y_eval = X[idx], y_true[idx]
    y_pred = model(X_eval)
    result = torch.mean((y_eval == torch.argmax(y_pred, dim=1)).float())
    model.train()
    return result

# From EECS598
def _extract_tensors(dset, num=None, x_dtype=torch.float32):
    """
    Extract the data and labels from a CIFAR10 dataset object and convert them to
    tensors.

    Input:
    - dset: A torchvision.datasets.CIFAR10 object
    - num: Optional. If provided, the number of samples to keep.
    - x_dtype: Optional. data type of the input image

    Returns:
    - x: `x_dtype` tensor of shape (N, 3, 32, 32)
    - y: int64 tensor of shape (N,)
    """
    x = torch.tensor(dset.data, dtype=x_dtype).permute(0, 3, 1, 2).div_(255)
    y = torch.tensor(dset.targets, dtype=torch.int64)
    if num is not None:
        if num <= 0 or num > x.shape[0]:
            raise ValueError(
                "Invalid value num=%d; must be in the range [0, %d]" % (num, x.shape[0])
            )
        x = x[:num].clone()
        y = y[:num].clone()
    return x, y