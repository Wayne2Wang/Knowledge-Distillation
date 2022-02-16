import os
import torch
import torchvision
import torchvision.transforms as transforms

import utils



def CIFAR10(root='data/CIFAR10/', flatten=True, validation_ratio=0.2,device='cpu',dtype=torch.float32, verbose=True):
    """
    Load CIFAR10 from given directory, normalize the data, split the training data to training and validation

    Input:
    - root: the directory to store or storing dataset
    - flatten: if True, return the flattend images
    - device: the device to store and process images
    - dtype: read the images as dtype
    - verbose: if True, print necessary information

    Returns:
    - data_dict: a dictionary storing the data
    - unnormalize: a dictionary storing the information needed for visualization
    """

    # Download and read dataset
    download = not os.path.isdir(root+"cifar-10-batches-py")
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,download=download)
    testset = torchvision.datasets.CIFAR10(root=root, train=False,download=download)

    if verbose and not download:
        print('Loaded data from {}'.format(root+"cifar-10-batches-py"))
    
    # Extract tensors from dataset
    num_train, num_test = len(trainset), len(testset)
    X_train, y_train = utils._extract_tensors(trainset,num_train,dtype)
    X_test, y_test = utils._extract_tensors(testset,num_test,dtype)

    # Move to the right device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Normalize the data: subtract the mean RGB (zero mean)
    mean_image = X_train.mean(dim=(0, 2, 3), keepdim=True)
    X_train -= mean_image
    X_test -= mean_image

    # Reshape the image data into rows
    if flatten:
      X_train = X_train.reshape(X_train.shape[0], -1)
      X_test = X_test.reshape(X_test.shape[0], -1)

    # Split the trainset to train and validation set randomly
    num_train2 = int(num_train * (1.0 - validation_ratio))
    num_validation = num_train - num_train2
    shuffler = torch.randperm(num_train)
    X_train_shuffled = X_train[shuffler]
    y_train_shuffled = y_train[shuffler]

    # return the dataset
    data_dict = {}
    data_dict["X_val"] = X_train_shuffled[num_train2 : num_train2 + num_validation]
    data_dict["y_val"] = y_train_shuffled[num_train2 : num_train2 + num_validation]
    data_dict["X_train"] = X_train_shuffled[0:num_train2]
    data_dict["y_train"] = y_train_shuffled[0:num_train2]

    data_dict["X_test"] = X_test
    data_dict["y_test"] = y_test

    # mean image needed for  visualization
    unnormalize = {'mean': mean_image}
    return data_dict, unnormalize


def imagenet1k():
    pass


def main():
    CIFAR10('data/')

if __name__ == '__main__':
    main()