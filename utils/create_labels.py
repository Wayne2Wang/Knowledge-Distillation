import os
import pandas as pd

"""
A script that converts imagenet label encoding to caffe style encoding
    and creates a label txt for training data

imagenet_val_labels_dir: imagenet encoded validation labels
imagenet_label_names_dir: imagenet label-name mapping
name_dir: all category names
"""

if __name__ == '__main__':

    root='data/ImageNet1k/'
    name_dir = root+'ILSVRC/Data/CLS-LOC/train/' # the folder that contrains subfolders for all classes
    imagenet_val_labels_dir = root+'ILSVRC/ImageSets/CLS-LOC/ILSVRC2012_validation_ground_truth.txt'
    imagenet_label_names_dir = root+'ILSVRC/ImageSets/CLS-LOC/ILSVRC2012_mapping.txt'

    names = os.listdir(name_dir)
    val_labels_imagenet = pd.read_csv(imagenet_val_labels_dir, delimiter=' ', header=None).to_numpy().reshape(-1)
    label_names = pd.read_csv(imagenet_label_names_dir, delimiter=' ', header=None)
    label_names = {idx:name[1] for idx, name in label_names.iterrows()}
    val_labels = [names.index(label_names[label-1]) for label in val_labels_imagenet]


    with open('val_labels.txt', 'w') as f:
        for item in val_labels:
            f.write("{}\n".format(item))

    # training set
    train_imageset_dir = root+'ILSVRC/ImageSets/CLS-LOC/train_cls.txt'
    names = os.listdir(name_dir)
    train_imageset = pd.read_csv(train_imageset_dir, delimiter=' ', header=None)
    train_labels = []
    for i in range(len(train_imageset)):
        image = train_imageset.iloc[i,0]
        name = image.split('/')[0]
        label = names.index(name)
        train_labels.append(label)

    with open('train_labels.txt', 'w') as f:
        for item in train_labels:
            f.write("{}\n".format(item))