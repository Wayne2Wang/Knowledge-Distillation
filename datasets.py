import os
import ast
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader



class ImageNet_dataset(Dataset):
    def __init__(self,root, dtype, train=True, flat=True, intensity=1, upscale=False, evalmode=False):
        self.root = root
        self.dtype = dtype
        self.flat = flat
        self.train = train
        # upscale -> upscales image dimension to a specified size
        self.upscale = upscale
        self.evalmode = evalmode
        
        # file directories
        # label: 0, name: n01440764, name_readable: tench
        train_imageset = root+'ILSVRC/ImageSets/CLS-LOC/train_cls.txt'
        train_image_dir = root+'ILSVRC/Data/CLS-LOC/train/'
        train_labels_dir = root+'ILSVRC/ImageSets/CLS-LOC/train_labels.txt'
        val_imageset = root+'ILSVRC/ImageSets/CLS-LOC/val.txt'
        val_image_dir = root+'ILSVRC/Data/CLS-LOC/val/'
        val_labels_dir = root+'ILSVRC/ImageSets/CLS-LOC/val_labels.txt'
        names_readable_dir = root+'ILSVRC/ImageSets/CLS-LOC/imagenet1000_clsidx_to_labels.txt'


        # Transformation and augmentation
        self.transforms = ImageNet_dataset.get_transformation(root)
        self.augmentations = ImageNet_dataset.get_augmentation(intensity) # intensity of augmentation. It is advised to keep it between 0 and 2

        # Read train/val data and labels
        if train:
            self.ImageSet = pd.read_csv(train_imageset, delimiter=' ', header=None)
            self.image_dir = train_image_dir
            self.labels = pd.read_csv(train_labels_dir, delimiter=' ', header=None).to_numpy().reshape(-1)
        else:
            self.ImageSet = pd.read_csv(val_imageset, delimiter=' ', header=None)
            self.image_dir = val_image_dir
            self.labels = pd.read_csv(val_labels_dir, delimiter=' ', header=None).to_numpy().reshape(-1)

        # Read label names and readable names
        self.names = os.listdir(train_image_dir)
        with open(names_readable_dir) as f:
            dict = f.read()
        self.names_readable = ast.literal_eval(dict)

    @staticmethod
    def get_transformation(root):
        if root.endswith('ImageNet1k/'):
            transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                            std = [ 0.229, 0.224, 0.225 ])])
        elif root.endswith('ImageNet64/'):
            # mean and std calculated on Imagenet64
            transform = transforms.Compose([
                        transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.4850, 0.4581, 0.4073],
                                            std = [0.2639, 0.2560, 0.2703])])
        else:
            raise Exception(root+' dataset not supported!')
        return transform

    @staticmethod
    def get_augmentation(intensity=1):
        # note that this only works for Tensor objects
        # It is advised to keep intensity between 0 and 2
        assert intensity >= 0, "Intensity should not be negative"
        augmentation = torch.nn.Sequential(
                        transforms.RandomAffine(degrees=(intensity*-30,intensity*30), translate=(intensity*0.1,intensity*0.1), 
                                                scale=((1-intensity*0.2),(1+intensity*0.2)), shear=intensity*30, 
                                                interpolation=transforms.InterpolationMode.BILINEAR
                                                ), # applies random affine transforoms (actually might be all we need)
                        transforms.RandomHorizontalFlip(p=intensity*0.25), # apply random horizontal flip with probability 0.25*intensity
                        transforms.RandomVerticalFlip(p=intensity*0.25),
                        transforms.RandomPerspective(distortion_scale=intensity*0.3, p=0.25) # apply perspective shift with probability 0.25
                        )
        return augmentation
    
    def __getitem__(self, idx):
        # Read image and label
        image_name = self.ImageSet.iloc[idx,0]
        img = Image.open(self.image_dir+image_name+'.jpeg').convert('RGB')
        label = self.labels[idx]

        # Perform transformation and augmentation
        if self.transforms:
            img = self.transforms(img)
        if self.upscale:
            upscaler = torch.nn.Upsample((self.upscale, self.upscale))
            img = upscaler(img)
        if self.augmentations and self.evalmode:
            img = self.augmentations(img)

        # flatten
        if self.flat:
            img = img.reshape(-1)

        return img, label
    
    def pred_to_name(self, pred):
        return self.label_names_readable[pred]
        
    
    def __len__(self):
        return self.ImageSet.shape[0]
    

class CIFAR10_dataset():
    def __init__(self,root, dtype, train=True, flat=True, intensity=1, upscale=False, evalmode=False):
        self.root = root
        self.dtype = dtype
        self.flat = flat
        self.train = train
        # upscale -> upscales image dimension to a specified size
        self.upscale = upscale
        self.evalmode = evalmode
        
        transform = self.get_transformation(evalmode, intensity)
        
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train,
                                                download=True, transform=transform)
        
        
        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    @staticmethod
    def get_transformation(evalmode, intensity):
        if evalmode == True:
            """
            transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],
                                                                  std = [0.2023, 0.1994, 0.201]),
                                             transforms.RandomAffine(degrees=(intensity*-30,intensity*30), translate=(intensity*0.1,intensity*0.1), 
                                                                     scale=((1-intensity*0.2),(1+intensity*0.2)), shear=intensity*30, 
                                                                     interpolation=transforms.InterpolationMode.BILINEAR
                                                                     ), # applies random affine transforoms (actually might be all we need)
                                             transforms.RandomHorizontalFlip(p=intensity*0.25), # apply random horizontal flip with probability 0.25*intensity
                                             transforms.RandomVerticalFlip(p=intensity*0.25),
                                             transforms.RandomPerspective(distortion_scale=intensity*0.3, p=0.25)
                                             ])
            """
            # changed to easier transform
            transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],
                                                                  std = [0.2023, 0.1994, 0.201]),
                                             transforms.RandomAffine(degrees=0, translate=(intensity*0.2,intensity*0.2),
                                                                                               shear=0,
                                                                     interpolation=transforms.InterpolationMode.NEAREST
                                                                     )#, # applies random affine transforoms (actually might be all we need)
                                             #transforms.RandomHorizontalFlip(p=intensity*0.25), # apply random horizontal flip with probability 0.25*intensity
                                             #transforms.RandomVerticalFlip(p=intensity*0.25)
                                             ])
        else:
            transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],
                                                                  std = [0.2023, 0.1994, 0.201])])
        return transform
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def pred_to_name(self, pred):
        return self.classes[pred]
        
    def __len__(self):
        return len(self.dataset)


class MNIST_dataset():
    def __init__(self,root, dtype, train=True, flat=True, intensity=1, upscale=False, evalmode=False):
        self.root = root
        self.dtype = dtype
        self.flat = flat
        self.train = train
        # upscale -> upscales image dimension to a specified size
        self.upscale = upscale
        self.evalmode = evalmode
        
        transform = self.get_transformation(evalmode, intensity)
        
        self.dataset = torchvision.datasets.MNIST(root=root, train=train,
                                                download=True, transform=transform)
        

    @staticmethod
    def get_transformation(evalmode, intensity):
        if evalmode == True:
            # changed to easier transform
            transform = transforms.Compose(
                                            [transforms.ToTensor(),
                                             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             transforms.Normalize((0.1307,), (0.3081,)),
                                             transforms.RandomAffine(degrees=0, translate=(intensity*0.2,intensity*0.2),
                                                                                               shear=0,
                                                                     interpolation=transforms.InterpolationMode.NEAREST
                                                                     )#, # applies random affine transforoms (actually might be all we need)
                                             #transforms.RandomHorizontalFlip(p=intensity*0.25), # apply random horizontal flip with probability 0.25*intensity
                                             #transforms.RandomVerticalFlip(p=intensity*0.25)
                                             ])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,)),])
        return transform
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label #img.repeat(3, 1, 1), label
        
    def __len__(self):
        return len(self.dataset)


def ImageNet(root='data/ImageNet1k/', flat=True, dtype=torch.float32, verbose=True, show=False, evalmode=False, intensity=1):
    trainset = ImageNet_dataset(root=root, train=True, dtype=dtype, flat=flat, evalmode=evalmode, intensity=intensity)
    valset = ImageNet_dataset(root=root, train=False, dtype=dtype, flat=flat, evalmode=evalmode, intensity=intensity)

    if verbose:
        print('Successfully loaded ImageNet from {}, image shape {}\n'.format(root, trainset[0][0].numpy().shape))
    
    # Show a random example
    if not flat and show:
        rand_int = torch.randint(len(trainset),(1,)).item()
        img, _ = trainset[rand_int] # 3xHxW
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

    return trainset, valset

def CIFAR(root='data/CIFAR10/', flat=False, dtype=torch.float32, verbose=True, show=False, evalmode=False, intensity=1):
    trainset = CIFAR10_dataset(root=root, train=True, dtype=dtype, flat=flat, evalmode=evalmode, intensity=intensity)
    valset = CIFAR10_dataset(root=root, train=False, dtype=dtype, flat=flat, evalmode=evalmode, intensity=intensity)
    
    if verbose:
        print('Successfully loaded CIFAR-10 from {}, image shape {}\n'.format(root, trainset[0][0].numpy().shape))
    
    # Show a random example
    if not flat and show:
        rand_int = torch.randint(len(trainset),(1,)).item()
        img, _ = trainset[rand_int] # 3xHxW
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

    return trainset, valset

def MNIST(root='data/MNIST/', flat=False, dtype=torch.float32, verbose=True, show=False, evalmode=False, intensity=1):
    trainset = MNIST_dataset(root=root, train=True, dtype=dtype, flat=flat, evalmode=evalmode, intensity=intensity)
    valset = MNIST_dataset(root=root, train=False, dtype=dtype, flat=flat, evalmode=evalmode, intensity=intensity)
    
    if verbose:
        print('Successfully loaded MNIST from {}, image shape {}\n'.format(root, trainset[0][0].numpy().shape))
    
    # Show a random example
    if not flat and show:
        rand_int = torch.randint(len(trainset),(1,)).item()
        img, _ = trainset[rand_int] # 3xHxW
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

    return trainset, valset

def main():
    #ImageNet(root='data/ImageNet64/',show=True, flat=False)
    CIFAR(show=True) # change root to datapath or CIFAR: 'D:/Research/Dataset/CIFAR10'

if __name__ == '__main__':
    main()
