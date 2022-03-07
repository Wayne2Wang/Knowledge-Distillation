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
    python_scripts_to_run = ['eval.py']
    args = [' --modelpath log/CIFAR10/MLP_drop_bnorm_56_model.pt --augment -1 --dataset CIFAR10 --root data/CIFAR10',
            ' --modelpath log/CIFAR10/MLP_drop_bnorm_56_model.pt --augment 0.1 --dataset CIFAR10 --root data/CIFAR10',
            ' --modelpath log/CIFAR10/MLP_drop_bnorm_56_model.pt --augment 0.2 --dataset CIFAR10 --root data/CIFAR10',
            ' --modelpath log/CIFAR10/MLP_drop_bnorm_56_model.pt --augment 0.4 --dataset CIFAR10 --root data/CIFAR10',
            ' --modelpath log/CIFAR10/MLP_drop_bnorm_56_model.pt --augment 0.8 --dataset CIFAR10 --root data/CIFAR10',
            ' --modelpath log/CIFAR10/MLP_drop_bnorm_56_model.pt --augment 1.6 --dataset CIFAR10 --root data/CIFAR10'
            ]
    for f in python_scripts_to_run:
        for arg in args:
            procs.append(f+arg)
            
elif mode=='train':
    python_scripts_to_run = ['train.py', 'train_kd.py']
    args = ['--model MLP_drop_bnorm --lr 1e-3 --bs 256 --hs 1024 512 256 128 64 32 --dataset CIFAR10 --verbose --epochs 60 --root "D:/Research/Dataset/CIFAR10"'
            ]
    args2 = ['--stdmodel MLP_drop_bnorm --lr 1e-3 --bs 256 --hs 1024 512 256 128 64 32 --dataset CIFAR10 --verbose --epochs 60 --root "D:/Research/Dataset/CIFAR10"'
            ]
    
    procs = [python_scripts_to_run[0] + " " + args[0],
            python_scripts_to_run[1] + " " + args2[0]]


        
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