# Knowledge-Distillation


List of pretrained CIFAR Models (by chenyaofo):
https://github.com/chenyaofo/pytorch-cifar-models

To train the models on ImageNet: 

1. download ImageNet1k(ILSVRC2012) to root/data/ImageNet1k
2. copy val_labels.txt and train_labels.txt into the ImageSet folder


To set up the right environment, run the following command
```
pip install -r requirements.txt
```

Train example arguments
```
train.py --model MLP --lr 1e-3 --bs 128 --dataset CIFAR10 --verbose --epochs 10 --root "D:/Research/Dataset/CIFAR10"
```
Train KD example arguments
```
train_kd.py --stdmodel MLP --tchmodel cifar10_resnet20 --lr 1e-3 --bs 128 --dataset CIFAR10 --verbose --epochs 10 --root "D:/Research/Dataset/CIFAR10"
```
Evaluate example arguments
```
eval.py --modelpath log/CIFAR10/MLP_drop_CifarResNet_1_model_KD.pt --augment --dataset CIFAR10 --num_batches 1000 --root "G:/My Drive/DL Data/"
```

To resume training, run
```
python train.py --load_model PATH_TO_MODEL
```

To view training stats, run
```
tensorboard --logdir log/DATASET
```
