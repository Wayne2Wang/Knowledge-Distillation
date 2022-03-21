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
    args = [#[' --modelpath log/CIFAR10/MLP_drop_bnorm_56_model.pt --augment -1 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm_eval.txt',
            #' --modelpath log/CIFAR10/MLP_drop_bnorm_56_model.pt --augment 0.1 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm_eval.txt',
            #' --modelpath log/CIFAR10/MLP_drop_bnorm_56_model.pt --augment 0.2 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm_eval.txt',
            #' --modelpath log/CIFAR10/MLP_drop_bnorm_56_model.pt --augment 0.4 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm_eval.txt',
            #' --modelpath log/CIFAR10/MLP_drop_bnorm_56_model.pt --augment 0.8 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm_eval.txt',
            #' --modelpath log/CIFAR10/MLP_drop_bnorm_56_model.pt --augment 1.6 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_bnorm_eval.txt'
            #],
            [' --modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment -1 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_brnom_KD_eval.txt',
            ' --modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment 0.1 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_brnom_KD_eval.txt',
            ' --modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment 0.2 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_brnom_KD_eval.txt',
            ' --modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment 0.4 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_brnom_KD_eval.txt',
            ' --modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment 0.8 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_brnom_KD_eval.txt',
            ' --modelpath log/CIFAR10/MLP_drop_bnorm_CifarResNet_59_model_KD.pt --augment 1.6 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/MLP_drop_brnom_KD_eval.txt'
            ]#,
            #[' --modelpath resnet --augment -1 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/resnet_eval.txt',
            #' --modelpath resnet --augment 0.1 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/resnet_eval.txt',
            #' --modelpath resnet --augment 0.2 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/resnet_eval.txt',
            #' --modelpath resnet --augment 0.4 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/resnet_eval.txt',
            #' --modelpath resnet --augment 0.8 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/resnet_eval.txt',
            #' --modelpath resnet --augment 1.6 --dataset CIFAR10 --root data/CIFAR10 --outfilepath log/CIFAR10/resnet_eval.txt'
            #]
            ]
    for i in range(len(python_scripts_to_run)):
        for arg in args[i]:
            procs.append(python_scripts_to_run[i]+arg)
            
elif mode=='train':
    python_scripts_to_run = ['train_kd.py', 'train.py']
    args = [[' --stdmodel MLP_drop_bnorm --lr 1e-3 --bs 256 --hs 512 256 128 --dataset CIFAR10 --alpha=0.3 --temp 1 --verbose --epochs 60 --root "D:/Research/Dataset/CIFAR10"'],
             [' --model MLP_drop_bnorm, --lr 1e-3, --bs 256, --hs 512 256 128, --dataset CIFAR10 --verbose --epochs 60 --root "D:/Research/Dataset/CIFAR10"']
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