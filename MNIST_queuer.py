"""
Script queuer

Queues script so they run in sequentially without having to augment arguments multiple times.

Got it to work after using vscode
"""

import os
import subprocess
from time import sleep

#--modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment -1 --dataset CIFAR10 --root data/CIFAR10
mode = 'train'
procs = []
if mode=='eval':
    print("Current setting: translation only")
    python_scripts_to_run = ['eval.py']
    args = [
                [' --modelpath log/MNIST/MLP/CNN_MNIST/30_model_KD.pt --augment -1 --dataset MNIST --root data/MNIST --outfilepath log/MNIST/MLP_KD_NOAUG_eval.txt',
                 ' --modelpath log/MNIST/MLP/CNN_MNIST/30_model_KD.pt --augment 1 --dataset MNIST --root data/MNIST --outfilepath log/MNIST/MLP_KD_AUG_eval.txt',
                 ' --modelpath log/MNIST/MLP/20_model.pt --augment -1 --dataset MNIST --root data/MNIST --outfilepath log/MNIST/MLP_NOAUG_eval.txt',
                 ' --modelpath log/MNIST/MLP/20_model.pt --augment 1 --dataset MNIST --root data/MNIST --outfilepath log/MNIST/MLP_AUG_eval.txt',
                ]
            ]
    for i in range(len(python_scripts_to_run)):
        for arg in args[i]:
            procs.append(python_scripts_to_run[i]+arg)
            
elif mode=='train':
    python_scripts_to_run = ['train_kd.py', 'train.py']
    args = [
                [' --stdmodel MLP --dataset MNIST --epochs 30 --root data/MNIST'], # train loss 0.001999, train_acc1 0.997000, train_acc5 1.000000, val_acc1 0.985900, val_acc5 0.999800
                [' --model MLP --dataset MNIST --epochs 20 --root data/MNIST'] # train loss 0.000102, train_acc1 0.996050, train_acc5 0.999983, val_acc1 0.980500, val_acc5 0.999400
            ]
    for i in range(len(python_scripts_to_run)):
        for arg in args[i]:
            procs.append(python_scripts_to_run[i]+arg)


def stream_process(process):
    go = process.poll() is None
    for line in process.stdout:
        print(line)
    return go


for proc in procs:
    print("\nArgument: {}".format(proc))
    os.system("python " + proc)
