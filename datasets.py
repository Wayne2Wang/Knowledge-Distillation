import os
import torch
import ast
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt


class ImageNet_dataset(Dataset):
    def __init__(self,root, dtype, train=True, flat=True, intensity=1, upscale=False, dataAug=False):
        self.root = root
        self.dtype = dtype
        self.flat = flat
        self.train = train
        # upscale -> upscales image dimension to a specified size
        self.upscale = upscale
        self.dataAug = dataAug
        
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
        if self.augmentations and self.dataAug:
            img = self.augmentations(img)

        # flatten
        if self.flat:
            img = img.reshape(-1)

        return img, label
    
    def pred_to_name(self, pred):
        return self.label_names_readable[pred]
        
    
    def __len__(self):
        return self.ImageSet.shape[0]

class ImageNet_Pickle(torch.utils.data.IterableDataset):
    def __init__(self, root, dtype, train=True, flat=False, intensity=1, imgsize=64, dataAug=False):
        self.root = root # root to dataset
        self.dtype = dtype
        self.train = train
        self.flat = flat
        self.imgsize = imgsize # imgsize (H == W)
        self.dataAug = dataAug
        self.dataLen = -1 # placeholder for length

        # Transformation and augmentation
        self.transforms = ImageNet_Pickle.get_transformation(dtype)
        self.augmentations = ImageNet_Pickle.get_augmentation(intensity) # intensity of augmentation. It is advised to keep it between 0 and 2


    @staticmethod
    def get_transformation(dtype):
        transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(dtype)])
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
    
    def __iter__(self):
        ## Generator-style dataloader
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or worker_info.num_workers != len(self.paths):
           raise ValueError("Number of workers doesn't match number of files.")  
        yield from self.read_data() # call sub-generator
    
    def __len__(self):
        if self.dataLen < 0:
            # initialize dataLen variable
            if self.train:
                num_files = 10
                for i in range(num_files):
                  # load 10 training data (might have to change this part if you modify the batch train data itself)
                  datapath = os.path.join(self.root, 'train_data_batch_')
                  dataset = np.load(datapath + str(i+1), mmap_mode='r+', allow_pickle=True) # load data and map it to memory (direct loading requires too much memory)
                  tempdata = dataset['data']
                  self.dataLen += tempdata.shape[0]
            else:
                datapath = os.path.join(self.root, 'val_data')
                dataset = np.load(datapath, mmap_mode='r', allow_pickle=True) # load data and map it to memory (direct loading requires too much memory)
                tempdata = dataset['data']
                self.dataLen = tempdata.shape[0]
            del dataset, tempdata
        return self.dataLen
    
    def read_data(self):
        # Read train/val data and labels
        """
        params
        :mode = 'gen' if generating, 'len' if calculating length
        """
        # generates processed data
        if self.train:
          num_files = 10
          for i in range(num_files):
            # load 10 training data (might have to change this part if you modify the batch train data itself)
            datapath = os.path.join(self.root, 'train_data_batch_')
            dataset = np.load(datapath + str(i+1), mmap_mode='r', allow_pickle=True) # load data and map it to memory (direct loading requires too much memory)
            yield from self.process_data(dataset) # call sub-sub-generator
        else:
          datapath = os.path.join(self.root, 'val_data')
          dataset = np.load(datapath, mmap_mode='r', allow_pickle=True) # load data and map it to memory (direct loading requires too much memory)
          yield from self.process_data(dataset)
            
            
    
    def process_data(self, dataset):
        # Processes and normalizes image data
        ImageSet = dataset['data']
        labels = np.array(dataset['labels'])
        meanImg = dataset['mean']
        meanImg /= np.float32(255)
        
        for j in range(ImageSet.shape[0]):
            # Read current image and label
            img = ImageSet[j]
            label = labels[j]-1 # 1-indexing to 0-indexing
            
            # Normalize
            img = img / np.float32(255)
            img -= meanImg
            
            # Reshape
            if not self.flat:
              img = img.reshape(3, self.imgsize, self.imgsize) #(3, 64, 64) if default
              
            # Perform transformation and augmentation
            if self.transforms:
                img = self.transforms(img)
            if self.augmentations and self.dataAug:
                img = self.augmentations(img)
                
            yield zip(img, label)
        

def ImageNet(root='data/ImageNet1k/', flat=True, dtype=torch.float32, 
             verbose=True, show=False, intensity=1,  dataAug=False, img_generator=True, imgsize=64):
    if img_generator:
        trainset = ImageNet_Pickle(root=root, dtype=dtype, train=True, flat=flat, intensity=intensity, imgsize=imgsize, dataAug=dataAug)
        valset = ImageNet_Pickle(root=root, dtype=dtype, train=False, flat=flat, intensity=intensity, imgsize=imgsize, dataAug=dataAug)
    else:
        trainset = ImageNet_dataset(root=root, train=True, dtype=dtype, flat=flat, dataAug=dataAug)
        valset = ImageNet_dataset(root=root, train=False, dtype=dtype, flat=flat, dataAug=dataAug)

    if verbose:
        if not img_generator:
            print('Successfully loaded ImageNet from {}, image shape {}\n'.format(root, trainset[0][0].numpy().shape))
        else:
            print('Successfully loaded ImageNet{} from {}, num_images(train, val): ({}, {})'.format(imgsize, root, len(trainset), len(valset)))
    
    # Show a random example
    if not flat and show and not img_generator:
        rand_int = torch.randint(len(trainset),(1,)).item()
        img, _ = trainset[rand_int] # 3xHxW
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

    return trainset, valset


def main():
    ImageNet(root='data/ImageNet64/',show=True, flat=False)

if __name__ == '__main__':
    main()
