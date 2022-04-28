"""
Script queuer

Queues script so they run in sequentially without having to augment arguments multiple times.

Got it to work after using vscode
"""

import os
import subprocess
from time import sleep

#--modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment -1 --dataset CIFAR10 --root data/CIFAR10
mode = 'eval'
procs = []
if mode=='eval':
    print("Current setting: translation only")
    python_scripts_to_run = ['eval.py']
    args = [[#' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_20_model.pt --augment -1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_result.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_20_model.pt --augment 0.1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_result.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_20_model.pt --augment 0.2 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_result.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_20_model.pt --augment 0.4 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_result.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_20_model.pt --augment 0.8 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_result.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_20_model.pt --augment 1.6 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_result.csv',

             ' --modelpath assets/CNN_JP4_30_model.pt --augment -1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath assets/CNN_JP4_30_model_result.csv',
             ' --modelpath assets/CNN_JP4_30_model.pt --augment 0.1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath assets/CNN_JP4_30_model_result.csv',
             ' --modelpath assets/CNN_JP4_30_model.pt --augment 0.2 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath assets/CNN_JP4_30_model_result.csv',
             ' --modelpath assets/CNN_JP4_30_model.pt --augment 0.4 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath assets/CNN_JP4_30_model_result.csv',
             ' --modelpath assets/CNN_JP4_30_model.pt --augment 0.8 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath assets/CNN_JP4_30_model_result.csv',
             ' --modelpath assets/CNN_JP4_30_model.pt --augment 1.6 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath assets/CNN_JP4_30_model_result.csv',

             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h2_20_model_KD.pt --augment -1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h2_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h2_20_model_KD.pt --augment 0.1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h2_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h2_20_model_KD.pt --augment 0.2 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h2_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h2_20_model_KD.pt --augment 0.4 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h2_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h2_20_model_KD.pt --augment 0.8 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h2_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h2_20_model_KD.pt --augment 1.6 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h2_20_model_KD.csv',

             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_20_model_KD.pt --augment -1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_20_model_KD.pt --augment 0.1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_20_model_KD.pt --augment 0.2 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_20_model_KD.pt --augment 0.4 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_20_model_KD.pt --augment 0.8 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_20_model_KD.pt --augment 1.6 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_20_model_KD.csv',

             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_20_model_KD.pt --augment -1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_20_model_KD.pt --augment 0.1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_20_model_KD.pt --augment 0.2 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_20_model_KD.pt --augment 0.4 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_20_model_KD.pt --augment 0.8 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_20_model_KD.pt --augment 1.6 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_t1.0_20_model_KD.pt --augment -1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_t1.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_t1.0_20_model_KD.pt --augment 0.1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_t1.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_t1.0_20_model_KD.pt --augment 0.2 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_t1.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_t1.0_20_model_KD.pt --augment 0.4 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_t1.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_t1.0_20_model_KD.pt --augment 0.8 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_t1.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_t1.0_20_model_KD.pt --augment 1.6 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h1_t1.0_20_model_KD.csv',

             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t7.0_20_model_KD.pt --augment -1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t7.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t7.0_20_model_KD.pt --augment 0.1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t7.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t7.0_20_model_KD.pt --augment 0.2 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t7.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t7.0_20_model_KD.pt --augment 0.4 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t7.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t7.0_20_model_KD.pt --augment 0.8 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t7.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t7.0_20_model_KD.pt --augment 1.6 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t7.0_20_model_KD.csv',

             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t21.0_20_model_KD.pt --augment -1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t21.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t21.0_20_model_KD.pt --augment 0.1 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t21.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t21.0_20_model_KD.pt --augment 0.2 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t21.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t21.0_20_model_KD.pt --augment 0.4 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t21.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t21.0_20_model_KD.pt --augment 0.8 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t21.0_20_model_KD.csv',
             #' --modelpath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t21.0_20_model_KD.pt --augment 1.6 --dataset CIFAR10 --root D:/Research/Dataset/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm3/MLP_drop_bnorm3_CNN_JP4_h0_t21.0_20_model_KD.csv',

            ],
            #[' --modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment -1 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_brnom_KD_eval.txt',
            #' --modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment 0.1 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_brnom_KD_eval.txt',
            #' --modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment 0.2 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_brnom_KD_eval.txt',
            #' --modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment 0.4 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_brnom_KD_eval.txt',
            #' --modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment 0.8 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_brnom_KD_eval.txt',
            #' --modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment 1.6 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_brnom_KD_eval.txt'
            #]#,
            #[
             #' --modelpath log/MNIST/MLP_drop_bnorm2_kd_featurematching_hook1/CNN_JP2/30_model_KD.pt --dataset MNIST_C --root D:/Research/Dataset/MNIST-C/mnist_c --augtype translate',
             #' --modelpath log/MNIST/MLP_drop_bnorm2_kd_featurematching_hook1/CNN_JP2/30_model_KD.pt --dataset MNIST_C --root D:/Research/Dataset/MNIST-C/mnist_c --augtype scale',
             #' --modelpath log/MNIST/MLP_drop_bnorm2_kd_featurematching_hook1/CNN_JP2/30_model_KD.pt --dataset MNIST --root D:/Research/Dataset/MNIST',
             #' --modelpath log/MNIST/MLP_drop_bnorm2_kd_featurematching_hook1/30_model.pt --dataset MNIST_C --root D:/Research/Dataset/MNIST-C/mnist_c --augtype translate',
             #' --modelpath log/MNIST/MLP_drop_bnorm2_kd_featurematching_hook1/30_model.pt --dataset MNIST_C --root D:/Research/Dataset/MNIST-C/mnist_c --augtype scale',
             #' --modelpath log/MNIST/MLP_drop_bnorm2_kd_featurematching_hook1/30_model.pt --dataset MNIST --root D:/Research/Dataset/MNIST',
             #' --modelpath log/MNIST/MLP_drop_bnorm/CNN_MNIST/60_model_KD.pt --dataset MNIST_C --root D:/Research/Dataset/MNIST-C/mnist_c --augtype translate',
             #' --modelpath log/MNIST/MLP_drop_bnorm/CNN_MNIST/60_model_KD.pt --dataset MNIST_C --root D:/Research/Dataset/MNIST-C/mnist_c --augtype scale',
             #' --modelpath log/MNIST/MLP_drop_bnorm/CNN_MNIST/60_model_KD.pt --dataset MNIST --root D:/Research/Dataset/MNIST',
             #' --modelpath log/MNIST/MLP_drop_bnorm2_alpha0.3_normal/MLP_drop_bnorm2_CNN_JP2_30_model_KD.pt --dataset MNIST_C --root D:/Research/Dataset/MNIST-C/mnist_c --augtype translate',
             #' --modelpath log/MNIST/MLP_drop_bnorm2_alpha0.3_normal/MLP_drop_bnorm2_CNN_JP2_30_model_KD.pt --dataset MNIST_C --root D:/Research/Dataset/MNIST-C/mnist_c --augtype scale',
             #' --modelpath log/MNIST/MLP_drop_bnorm2_alpha0.3_normal/MLP_drop_bnorm2_CNN_JP2_30_model_KD.pt --dataset MNIST --root D:/Research/Dataset/MNIST'
            #]
            ]
    for i in range(len(python_scripts_to_run)):
        for arg in args[i]:
            procs.append(python_scripts_to_run[i]+arg)
            
elif mode=='train':
    python_scripts_to_run = [
                                #'train.py',
                                'train_kd.py',
                                ]
    args = [
            #[' --model CNN_JP4 --lr 1e-3 --bs 256 --reg 5e-3 --dataset CIFAR10 --verbose --epochs 5 --root D:/Research/Dataset/CIFAR10'],
            #[' --model MLP_drop_bnorm3 --lr 1e-3 --reg 1e-2 --bs 256 --hs 1024 512 256 128 --dataset CIFAR10 --verbose --epochs 20 --root D:/Research/Dataset/CIFAR10'],
            [
                ' --stdmodel MLP_drop_bnorm3 --lr 1e-3 --reg 1e-2 --bs 128 --hs 1024 512 256 128 --dataset CIFAR10 --alpha=0.1 --temp 1 --verbose --epochs 20 --root D:/Research/Dataset/CIFAR10 --hook=0 --hook_scale=1 --kl_scale=1',
                #' --stdmodel MLP_drop_bnorm3 --lr 1e-3 --reg 1e-2 --bs 128 --hs 1024 512 256 128 --dataset CIFAR10 --alpha=0.1 --temp 7 --verbose --epochs 20 --root D:/Research/Dataset/CIFAR10 --hook=0 --hook_scale=0 --kl_scale=1',
                #' --stdmodel MLP_drop_bnorm3 --lr 1e-3 --reg 1e-2 --bs 128 --hs 1024 512 256 128 --dataset CIFAR10 --alpha=0.1 --temp 21 --verbose --epochs 20 --root D:/Research/Dataset/CIFAR10 --hook=0 --hook_scale=0 --kl_scale=1',
            ],
            #[' --stdmodel MLP_drop_bnorm --lr 1e-3 --bs 256 --hs 512 256 128 --dataset CIFAR10 --alpha=0.3 --temp 1 --verbose --epochs 60 --root "D:/Research/Dataset/CIFAR10"']#,
            #' --stdmodel MLP_drop_bnorm --load_model log/CIFAR10/MLP_drop_bnorm_CifarResNet_56.pt --lr 1e-3 --bs 256 --hs 512 512 256 256 128 128 --dataset CIFAR10 --alpha=0.3 --temp 1 --verbose --epochs 0 --root "D:/Research/Dataset/CIFAR10"'
            ]
    for i in range(len(python_scripts_to_run)):
        for arg in args[i]:
            procs.append(python_scripts_to_run[i]+arg)


        
#results = []


def stream_process(process):
    go = process.poll() is None
    for line in process.stdout:
        print(line)
    return go


for proc in procs:
    #result = subprocess.Popen("python " + proc, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("\nArgument: {}".format(proc))
    #process = subprocess.Popen("python " + proc, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #subprocess.call("python " + proc, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #while stream_process(process):
    #    sleep(0.1)
    #output, error = process.communicate()
    #print(output)
    #process.terminate()
    #process.kill()
    #exec(open(proc).read())
    
    os.system("python " + proc)
    #results.append(result.communicate())
    #r = subprocess.check_output("python " + proc, shell=False)
    #print(r)
#print(results)