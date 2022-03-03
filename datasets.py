import os
import torch
import ast
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt


class ImageNet_dataset(Dataset):
    def __init__(self,root, dtype, train=True, flat=True, intensity=1, upscale=False):
        self.root = root
        self.dtype = dtype
        self.flat = flat
        self.train = train
        # upscale -> upscales image dimension to a specified size
        self.upscale = upscale
        
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
        #if self.augmentations and not self.train:
            #img = self.augmentations(img)

        # flatten
        if self.flat:
            img = img.reshape(-1)

        return img, label
    
    def pred_to_name(self, pred):
        return self.label_names_readable[pred]
        
    
    def __len__(self):
        return self.ImageSet.shape[0]


def ImageNet(root='data/ImageNet1k/', flat=True, dtype=torch.float32, verbose=True, show=False):
    trainset = ImageNet_dataset(root=root, train=True, dtype=dtype, flat=flat)
    valset = ImageNet_dataset(root=root, train=False, dtype=dtype, flat=flat)

    if verbose:
        print('Successfully loaded ImageNet from {}, image shape {}\n'.format(root, trainset[0][0].numpy().shape))
    
    # Show a random example
    if not flat and show:
        rand_int = torch.randint(len(trainset),(1,)).item()
        img, _ = trainset[rand_int] # 3xHxW
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

    return trainset, valset


def main():
    ImageNet(root='data/ImageNet64/',show=True, flat=False)

if __name__ == '__main__':
    main()
