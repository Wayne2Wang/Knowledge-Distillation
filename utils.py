import torch

def save_checkpoint(path_to_model, model, epoch, optimizer, criterion, best_loss, verbose=True):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion': criterion,
                'best_loss': best_loss
                }, path_to_model)
    if verbose:
        print('New best loss {:5f}! Saved the model to {}'.format(best_loss, path_to_model))

def load_checkpoint(path_to_model, verbose=True):
    checkpoint = torch.load(path_to_model)
    model = checkpoint['model_state_dict']
    epoch = checkpoint['epoch']
    optimizer = checkpoint['optimizer_state_dict']
    criterion = checkpoint['criterion']
    best_loss = checkpoint['best_loss']
    if verbose:
        print('Loaded checkpoint from {}'.format(path_to_model))
    return model, epoch, optimizer, criterion, best_loss

def eval_model_acc(model, X, y_true):
  model.eval()
  y_pred = model(X)
  result = torch.mean(y_true == torch.argmax(y_pred, dim=1))
  model.train()
  return result

def test():
    num = 0
    correct = 0

    '''
    num += 1
    model = torch.abs
    pred = torch.tensor([1,2,3,4,5,6,6])
    y_true = torch.tensor([1,2,3,4,5,6,6])
    if acc(model, pred, y_true) == 1:
        correct += 1
    '''

    return num, correct

if __name__ == '__main__':
    print('Running testcases')
    num, correct = test()
    print('Ran {} testcases, {} passed'.format(num, correct))

