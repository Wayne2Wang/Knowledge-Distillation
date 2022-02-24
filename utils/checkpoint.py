import os
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
    # Create directory if not exist
    path_to_model = 'log\\' + path_to_model
    dir_path = path_to_model[:path_to_model.rfind('\\')]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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
    best_loss = checkpoint['loss_value']
    train_acc = checkpoint['train_acc']
    val_acc = checkpoint['val_acc']
    if verbose:
        print('Loaded checkpoint from {}'.format(path_to_model))
        print('Previous loss {:5f}, train_acc {:5f}, val_acc {:5f}'.format(best_loss, train_acc, val_acc))
    return epoch, best_loss, train_acc, val_acc
