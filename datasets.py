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


class ImageNet1K_dataset(Dataset):
    def __init__(self,root, dtype, train=True, flat=True):
        self.root = root
        self.dtype = dtype
        self.flat = flat
        self.train = train

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
        self.transforms = ImageNet1K_dataset.get_transformation(flat)
        self.augmentations = ImageNet1K_dataset.get_augmentation()

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
    def get_transformation(flat):
        if flat:
            resize = 36
            center = 32
        else:
            resize = 256
            center = 224
        transform = transforms.Compose([
                        transforms.Resize(resize),
                        transforms.CenterCrop(center),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                            std = [ 0.229, 0.224, 0.225 ])])
        return transform

    @staticmethod
    def get_augmentation():
        # TODO: Implement augmentation for imagenet
        return None
    
    def __getitem__(self, idx):
        # Read image and label
        image_name = self.ImageSet.iloc[idx,0]
        img = Image.open(self.image_dir+image_name+'.jpeg').convert('RGB')
        label = self.labels[idx]

        # Perform transformation and augmentation
        if self.transforms:
            img = self.transforms(img)
        if self.augmentations:
            img = self.augmentations(img)

        # flatten
        if self.flat:
            img = img.reshape(-1)

        return img, label
    
    def pred_to_name(self, pred):
        return self.label_names_readable[pred]
        
    
    def __len__(self):
        return self.ImageSet.shape[0]


def ImageNet1k(root='data/ImageNet1k/', flat=True, dtype=torch.float32, verbose=True, show=False):
    trainset = ImageNet1K_dataset(root=root, train=True, dtype=dtype, flat=flat)
    valset = ImageNet1K_dataset(root=root, train=False, dtype=dtype, flat=flat)

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
    ImageNet1k(show=True, flat=False)

if __name__ == '__main__':
    main()