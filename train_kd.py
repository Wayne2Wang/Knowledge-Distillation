import torch
import torchvision
import tqdm
from torchsummary import summary
from datasets import ImageNet
from utils.checkpoint import load_checkpoint
from models import MLP, fit_model
import argparse
from eval import eval_acc
from torch.utils.data import DataLoader
import torchmetrics


def parse_arg():
	"""
	Parse the command line arguments
	"""
	parser = argparse.ArgumentParser(description='Arugments for fitting the feedforward model')
	parser.add_argument('--resnet', action='store_true', help='If True, train a resnet(for testing purpose)')
	parser.add_argument('--load_model', type=str, default='', help='Resume training from load_model if not empty')
	parser.add_argument('--dataset', type=str, default='ImageNet1k', help='Dataset used for training')
	parser.add_argument('--verbose', action='store_true', help='If True, print training progress')
	parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
	parser.add_argument('--bs', type=int, default=128, help='Batch size')
	parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
	parser.add_argument("--hs",  nargs="*",  type=int, default=[2000, 1000, 100], help='Hidden units')
	parser.add_argument('--save_every', type=int, default=1, help='Save every x epochs')
	# add arg for student and teacher model
	args = parser.parse_args()
	return args


def train_kd(teacherModel,
            studentModel,
			optimizerStudent,
			student_loss,
			divergence_loss,
			temp,
			alpha,
			device,
			data_loader,
            prev_epoch,
            train_size,
            epochs = 10,  
            batch_size = 64, 
            save_every = 5,
            verbose = False,
            ):
    
    for epoch in range(epochs):

        total_loss = 0
        real_epoch = prev_epoch+epoch+1

        studentModel.train()
        teacherModel.eval()
        
        for batch in tqdm(data_loader, ascii=True, desc='Epoch {}/{}'.format(real_epoch, prev_epoch+epochs)):
        
            # Prepare minibatch
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)

            real_epoch = prev_epoch
            model = model.to(device)
            metric1 = torchmetrics.Accuracy(top_k=1).to(device)
            metric5 = torchmetrics.Accuracy(top_k=5).to(device)
            
            # clear gradient
            optimizerStudent.zero_grad()

            # forward
            with torch.no_grad():
                teacher_preds = teacherModel(x_batch)

            student_preds = studentModel(x_batch)
            student_loss = student_loss(student_preds, y_batch)
                
            distillation_loss = divergence_loss(
                torch.nn.functional.softmax(student_preds / temp, dim=1),
                torch.nn.functional.softmax(teacher_preds / temp, dim=1)
            )

            # Compute train acc
            metric1.update(student_preds, y_batch)
            metric5.update(student_preds, y_batch)

            # Record loss   
            loss = alpha * student_loss + (1 - alpha) * distillation_loss
            loss_value = loss.item()
            total_loss += loss_value

            # backward
            
            loss.backward()

            optimizerStudent.step()

            # Evaluate this epoch
            total_loss /= train_size
            train_acc1 = metric1.compute()
            train_acc5 = metric5.compute()
            metric1.reset()
            metric5.reset()
            best_train_acc1 = train_acc1 if train_acc1 > best_train_acc1 else best_train_acc1
            best_train_acc5 = train_acc5 if train_acc5 > best_train_acc5 else best_train_acc5
    
    return train_acc1, train_acc5




def main():
	# Read the arguments
	args = parse_arg()
	load_model = args.load_model
	verbose = True #args.verbose
	dataset = args.dataset
	save_every = args.save_every
	resnet = args.resnet
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cpu'

	# Hyperparameters
	epochs = args.epochs
	batch_size = args.bs
	lr = args.lr
	hidden_sizes = args.hs
	temp = 7
	alpha = 0.3

	# Read data
	if dataset == 'ImageNet1k':	
		trainset, valset = ImageNet(root='data/ImageNet1k/',flat=(not resnet),verbose=verbose)
		output_size = 1000 # number of distinct labels
	elif dataset == 'ImageNet64':	
		trainset, valset = ImageNet(root='data/ImageNet64/',flat=(not resnet),verbose=verbose)
		output_size = 1000 # number of distinct labels
	else:
		raise Exception(dataset+' dataset not supported!')
	data = trainset, valset, dataset
	input_size = trainset[0][0].shape[0] # input dimensions

	trainset, valset, dataset = data
	train_size = len(trainset)
	train_loader = DataLoader(trainset, batch_size = batch_size, num_workers = 3, shuffle = False)
	val_loader = DataLoader(valset, batch_size = batch_size, num_workers = 3, shuffle = False)
	
	# Model initialization for teacher
	if resnet:
		teacherModel = torchvision.models.resnet18(pretrained=True)
	else:
		teacherModel = MLP(input_size, hidden_sizes, output_size)
		summary(teacherModel, (1,input_size), device='cpu')
	model_name = type(teacherModel).__name__
	teacherModel = teacherModel.to(device) # avoid different device error when resuming training
	prev_epoch = 0
	optimizer = torch.optim.AdamW(teacherModel.parameters(), lr=lr)
	criterion = torch.nn.CrossEntropyLoss()


	# Model initialization for student
	studentModel = MLP(input_size, hidden_sizes / 3, output_size)
	summary(studentModel, (1,input_size), device='cpu')
	
	model_name_student = type(studentModel).__name__
	studentModel = studentModel.to(device) # avoid different device error when resuming training
	prev_epoch = 0
	optimizerStudent = torch.optim.AdamW(studentModel.parameters(), lr=lr)
	student_loss = torch.nn.CrossEntropyLoss()
	divergence_loss = torch.nn.KLDivLoss(reduction="batchmean")
	optimizerStudent = torch.optim.AdamW(studentModel.parameters(), lr=lr)


	# Load previously trained model if specified
	if not load_model == '':
		prev_epoch, _, _, _ = load_checkpoint(teacherModel, optimizer, criterion, load_model, verbose)

	
	# Training
	if verbose:
		print('\nStart training {}: epoch={}, prev_epoch={}, batch_size={}, lr={}, save_every={} device={}'\
									.format(model_name, epochs, prev_epoch, batch_size, lr, save_every, device))

	# Train teacher Model

	best_loss, train_acc, val_acc, total_time = fit_model(teacherModel,
														  data, 
														  optimizer,
														  criterion,
														  device=device,
														  prev_epoch=prev_epoch,
														  epochs=epochs,  
														  batch_size=batch_size, 
														  save_every=save_every,
														  verbose=verbose)
	

	if verbose:
		print('\nTraining finished! \nTime: {:2f}, (best) train loss: {:5f}, train acc1: {:5f}, train acc5: {:5f}, val acc1: {:5f}, val acc5: {:5f}' \
														.format(total_time, best_loss, train_acc[0], train_acc[1], val_acc[0], val_acc[1]))

	# Train student model with soft labels from parent
	train_acc1, train_acc5 = train_kd(teacherModel,
									studentModel,
									optimizerStudent,
									student_loss,
									divergence_loss,
									temp = temp,
									alpha = alpha,
									epochs = epochs,
									device = device,
									data_loader = train_loader,
									prev_epoch=prev_epoch,
									train_size=train_size)
	
	# Evaluate accuracy of student model
	eval_acc(studentModel, val_loader, device, num_batches=None, verbose=False, mode='validation')										

if __name__ == '__main__':
	main()
