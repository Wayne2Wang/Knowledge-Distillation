import torch
import torchvision
from tqdm import tqdm
from torchsummary import summary
from datasets import *
from utils.checkpoint import load_checkpoint, save_checkpoint
from models import MLP
import argparse
from eval import eval_acc
from torch.utils.data import DataLoader
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import time
from utils.upscaler import ModelUpscaler

from experiment.design_JP import * # call your models here


def parse_arg():
    """
    Parse the command line arguments
    """
    parser = argparse.ArgumentParser(description='Arugments for fitting the feedforward model')
    #parser.add_argument('--resnet', action='store_true', help='If True, train a resnet(for testing purpose)')
    parser.add_argument('--stdmodel', type=str, default='MLP', help='Name of the student model you\'re going to train')
    parser.add_argument('--tchmodel', type=str, default='cifar10_resnet20', help='Name of the teacher model you\'re going to use')
    
    parser.add_argument('--load_model', type=str, default='', help='Resume training from load_model if not empty')
    parser.add_argument('--dataset', type=str, default='ImageNet1k', help='Dataset used for training')
    parser.add_argument('--verbose', action='store_true', help='If True, print training progress')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument("--hs",  nargs="*",  type=int, default=[2000, 1000, 100], help='Hidden units')
    parser.add_argument('--alpha', type=float, default=0.3, help='hyperparmeter')
    parser.add_argument('--temp', type=float, default=7, help='temperature')
    parser.add_argument('--save_every', type=int, default=1, help='Save every x epochs')
    
    parser.add_argument('--root', default='data/ImageNet64', help='Set root of dataset')
    parser.add_argument('--reg', default=1e-3, type=float, help="Specify the strength of the regularizer")
<<<<<<< Updated upstream
=======
    parser.add_argument('--thook', type=str, default='layername', help='Name of the teacher layer to match')
    parser.add_argument('--shook', type=str, default='layername', help='Name of the student layer to match')
    # MNIST-C Specific settings
    parser.add_argument('--augtype', type=str, default='translate', help='Data augmentation type for MNIST-C dataset. Will focus mostly on scale and translate')
>>>>>>> Stashed changes
    
    
    
    # add arg for student and teacher model
    args = parser.parse_args()
    return args

def train_kd(teacherModel,
            studentModel,
            data,
            optimizerStudent,
            student_loss,
            divergence_loss,
            temp,
            alpha,
            train_size,
            prev_epoch,
            epochs = 10,  
            batch_size = 64, 
            save_every = 5,
            verbose = False,
            teacherhook='layername', # name of layer to hook
            studenthook='layername', # name of layer to hook
            device='cpu'
            ):

    # Read data
    trainset, valset, dataset = data
    train_size = len(trainset)
    train_loader = DataLoader(trainset, batch_size = batch_size, num_workers = 3, shuffle = False)
    val_loader = DataLoader(valset, batch_size = batch_size, num_workers = 3, shuffle = False)
    model_name_student = type(studentModel).__name__
    model_name_teacher = type(teacherModel).__name__
    

    writer = SummaryWriter(log_dir='log/{}/{}_{}'.format(dataset, model_name_student, model_name_teacher))
    start_time = time.time()
    best_loss = float('inf')
    best_train_acc1 = 0
    best_train_acc5 = 0
    best_val_acc1 = 0
    best_val_acc5 = 0
    real_epoch = prev_epoch
    studentModel = studentModel.to(device)
    teacherModel=teacherModel.to(device)
    metric1 = torchmetrics.Accuracy(top_k=1).to(device)
    metric5 = torchmetrics.Accuracy(top_k=5).to(device)
    for epoch in range(epochs):

        total_loss = 0
        real_epoch = prev_epoch+epoch+1

        studentModel.train()
        teacherModel.eval()
        
        for batch in tqdm(train_loader, ascii=True, desc='Epoch {}/{}'.format(real_epoch, prev_epoch+epochs)):
        
            # Prepare minibatch
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            
            # clear gradient
            optimizerStudent.zero_grad()

            if teacherhook != 'layername': # if defined
                features={} # dict to store the intermediate activation
                def get_features(name):
                    """
                    extracts layer output of layername 'name'

                    inputs:
                    name: layername
                    """
                    def hook(model, input, output):
                        features[name] = output
                    return hook
                
                hteacher = teacherModel.cnn_layers.register_forward_hook(get_features('teacher_out')) # get hooked output
                hstudent = studentModel.bnorm[3].register_forward_hook(get_features('student_out')) # hardcoded!!!

            # forward
            with torch.no_grad():
                teacher_preds = teacherModel(x_batch)

            student_preds = studentModel(x_batch.reshape(x_batch.shape[0],-1))
            student_loss_value = student_loss(student_preds, y_batch)
            
            if teacherhook=='layername': # if not specified, do KL Div on output
                distillation_loss = divergence_loss(
                    torch.nn.functional.log_softmax(student_preds / temp, dim=1),
                    torch.nn.functional.softmax(teacher_preds / temp, dim=1)
                )
            else:
                distillation_loss = divergence_loss( # this should be L1 or L2 loss (MSE loss)
                    studentModel.relu(features['student_out']), features['teacher_out'].flatten(start_dim=1)
                )

            # Compute train acc
            metric1.update(student_preds, y_batch)
            metric5.update(student_preds, y_batch)

            # Record loss   
            loss = alpha * student_loss_value + (1 - alpha) * (temp**2) * distillation_loss
            loss_value = loss.item()
            total_loss += loss_value

            # backward
            
            loss.backward()

            optimizerStudent.step()

            hteacher.remove()
            hstudent.remove()

        # Evaluate this epoch
        total_loss /= train_size
        train_acc1 = metric1.compute()
        train_acc5 = metric5.compute()
        metric1.reset()
        metric5.reset()
        val_acc1, val_acc5 = eval_acc(studentModel, val_loader, device) 

        # update stats
        best_train_acc1 = train_acc1 if train_acc1 > best_train_acc1 else best_train_acc1
        best_train_acc5 = train_acc5 if train_acc5 > best_train_acc5 else best_train_acc5
        best_val_acc1 = val_acc1 if val_acc1 > best_val_acc1 else best_val_acc1
        best_val_acc5 = val_acc5 if val_acc5 > best_val_acc5 else best_val_acc5
        best_loss = total_loss if total_loss < best_loss else best_loss

        # log training stats
        writer.add_scalar('Loss/train', total_loss, real_epoch)
        writer.add_scalar('Accuracy1/train', train_acc1, real_epoch)
        writer.add_scalar('Accuracy5/train', train_acc5, real_epoch)
        writer.add_scalar('Accuracy1/val', val_acc1, real_epoch)
        writer.add_scalar('Accuracy5/val', val_acc5, real_epoch)

        if verbose:
            print('train loss {:5f}, train_acc1 {:5f}, train_acc5 {:5f}, val_acc1 {:5f}, val_acc5 {:5f}, time {:2f}s'.format(total_loss, train_acc1, train_acc5, val_acc1, val_acc5, time.time()-start_time))

        train_acc = train_acc1, train_acc5
        val_acc = val_acc1, val_acc5
        if (epoch+1)%save_every == 0:
            save_checkpoint('{}/{}_{}_{}.pt'.format(dataset, model_name_student, model_name_teacher, real_epoch),studentModel, real_epoch,\
             optimizerStudent, student_loss, total_loss, train_acc, val_acc, verbose=verbose)


    if not epochs%save_every == 0:
        save_checkpoint('{}/{}_{}_{}.pt'.format(dataset, model_name_student,model_name_teacher, real_epoch),studentModel, real_epoch,\
             optimizerStudent, student_loss, total_loss, train_acc, val_acc, verbose=verbose)
    
    ### Save whole model (no need to redefine it -- for evaluation purposes)
    torch.save(studentModel, 'log/{}/{}_{}_{}_model_KD.pt'.format(dataset, model_name_student,model_name_teacher, real_epoch))
    
    total_time = time.time() - start_time
    best_train_acc = best_train_acc1, best_train_acc5
    best_val_acc = best_val_acc1, best_val_acc5
    
    return best_loss, best_train_acc, best_val_acc, total_time



def mainkd(
    load_model,
    verbose,
    dataset,
    save_every,
    student_name,
    teacher_name,
    root,
    epochs,
    batch_size,
    lr,
    hidden_sizes,
    temp,
    alpha,
    regstr,
    augtype,
    teacherhook='layername',
    studenthook='layername'
):
    
    std_model_method = globals()[student_name] # call function with 'modelname' name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cpu'
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    
<<<<<<< Updated upstream
    # Hyperparameters
    epochs = args.epochs
    batch_size = args.bs
    lr = args.lr
    hidden_sizes = args.hs
    temp = args.temp
    alpha = args.alpha
    regstr = args.reg

    # Read data
=======
    # Read data and initialize model
>>>>>>> Stashed changes
    if dataset == 'ImageNet1k':    
        trainset, valset = ImageNet(root=root,flat=(False if modelname=='resnet' else True),verbose=verbose)
        output_size = 1000 # number of distinct labels
        input_size = trainset[0][0].reshape(-1).shape[0] # input dimensions
    elif dataset == 'ImageNet64':    
        trainset, valset = ImageNet(root=root,flat=(False if modelname=='resnet' else True),verbose=verbose)
        output_size = 1000 # number of distinct labels
        input_size = trainset[0][0].reshape(-1).shape[0] # input dimensions
    elif dataset == 'CIFAR10':
        trainset, valset = CIFAR(root=root, flat=False, verbose=verbose)
        output_size = 10
        input_size = trainset[0][0].shape
<<<<<<< Updated upstream
=======

        teacherModel = torch.hub.load("chenyaofo/pytorch-cifar-models", teacher_name, pretrained=True)
        studentModel = std_model_method(input_size, hidden_sizes, output_size).to(device)

    elif dataset == 'MNIST':
        trainset, valset = MNIST(root=root, flat=False, verbose=verbose)
        output_size = 10
        input_size = trainset[0][0].shape

        # Here teacher model is a vanilla CNN trained on MNIST for 10 epochs, to be improved
        #teacherModel = torch.load('assets/CNN_MNIST_10_model.pt')
        teacherModel = torch.load('log/MNIST/CNN_JP2/CNN_JP2_30_model.pt')
        studentModel = std_model_method(input_size, hidden_sizes, output_size).to(device)
    elif dataset == 'MNIST_C':
        trainset, valset = MNIST_C(root=root, verbose=verbose, type=augtype)
        output_size = 10
        input_size = trainset[0][0].shape

        teacherModel = torch.load('assets/60_model.pt')
        studentModel = std_model_method(input_size, hidden_sizes, output_size).to(device)
>>>>>>> Stashed changes
    else:
        raise Exception(dataset+' dataset not supported!')
    data = trainset, valset, dataset
    train_size = len(trainset)
    
    
    # Model initialization for teacher
    #teacherModel = torchvision.models.resnet50(pretrained=True)
    teacherModel = torch.hub.load("chenyaofo/pytorch-cifar-models", teacher_name, pretrained=True)
    model_name_teacher = type(teacherModel).__name__
    #teacherModel = ModelUpscaler(teacherModel, 224)


    # Model initialization for student
    studentModel = std_model_method(input_size, hidden_sizes, output_size)
    studentModel = studentModel.to(device) # avoid different device error when resuming training
    model_name_student = type(studentModel).__name__
    summary(studentModel, (1,) + tuple(input_size), device=device_name)
    
    prev_epoch = 0
    optimizerStudent = torch.optim.AdamW(studentModel.parameters(), lr=lr, weight_decay=regstr)
    student_loss = torch.nn.CrossEntropyLoss()
    if teacherhook == 'layername':
        divergence_loss = torch.nn.KLDivLoss(reduction="batchmean")
    else:
        divergence_loss = torch.nn.MSELoss()

    # Load previously trained model if specified
    if not load_model == '':
        prev_epoch, _, _, _ = load_checkpoint(studentModel, optimizerStudent, student_loss, load_model, verbose)

    
    # Training
    if verbose:
        print('\nStart training {} from teacher {}: epoch={}, prev_epoch={}, batch_size={}, lr={}, alpha={}, temp={}, save_every={} device={}'\
                                    .format(model_name_student, model_name_teacher, epochs, prev_epoch, batch_size, lr, alpha, temp, save_every, device))

    

    # Train student model with soft labels from parent
    best_loss, train_acc, val_acc, total_time = train_kd(teacherModel,
                                    studentModel,
                                    data,
                                    optimizerStudent,
                                    student_loss,
                                    divergence_loss,
                                    temp,
                                    alpha,
                                    train_size,
                                    prev_epoch,
                                    epochs = epochs,
                                    batch_size=batch_size,
                                    save_every=save_every,
                                    device = device,
                                    verbose = verbose,
                                    teacherhook=teacherhook,
                                    studenthook=studenthook
                                    )

    if verbose:
        print('\nTraining finished! \nTime: {:2f}, (best) train loss: {:5f}, train acc1: {:5f}, train acc5: {:5f}, val acc1: {:5f}, val acc5: {:5f}' \
                                                        .format(total_time, best_loss, train_acc[0], train_acc[1], val_acc[0], val_acc[1]))
    
                                    

if __name__ == '__main__':
    # Read the arguments
    args = parse_arg()
    load_model = args.load_model
    verbose = True #args.verbose
    dataset = args.dataset
    save_every = args.save_every
    
    student_name = args.stdmodel
    teacher_name = args.tchmodel

    root = args.root

    # Hyperparameters
    epochs = args.epochs
    batch_size = args.bs
    lr = args.lr
    hidden_sizes = args.hs
    temp = args.temp
    alpha = args.alpha
    regstr = args.reg
    augtype = args.augtype

    teacherhook = args.thook
    studenthook = args.shook

    custom_args = False # custom args for debugging
    if custom_args:
        load_model = ''
        verbose = True
        dataset = 'MNIST'
        save_every = 1

        student_name = 'MLP_drop_bnorm2' #MLP_drop_bnorm2
        teacher_name = 'CNN_JP2' #CNN_JP2
        root = 'D:/Research/Dataset/MNIST'

        epochs = 30
        batch_size = 64
        lr = 5e-4
        hidden_sizes = [256, 256, 128, 128]
        temp = 10**0.5
        alpha = 0.3
        regstr=1e-3
        augtype='translate'
        
        teacherhook = 'cnn_layers'
        studenthook = 'fc'


    mainkd( # pass to main ftn
        load_model, verbose, dataset, save_every,
        student_name, teacher_name, root, epochs,
        batch_size, lr, hidden_sizes, temp, alpha, regstr, augtype,
        teacherhook, studenthook
    )
