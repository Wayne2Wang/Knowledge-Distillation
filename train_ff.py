import math
import time
import torch
from torchsummary import summary
from data import CIFAR10


import utils
from models import feed_forward, fit_ff
import argparse


def parse_arg():
	"""
		Parse the command line arguments
	"""
	parser = argparse.ArgumentParser(description='Arugments for fitting the feedforward model')
	parser.add_argument('--load_model', type=str, default='', help='Resume training from load_model if not empty')
	parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset used for training')
	parser.add_argument('--verbose', action='store_true', help='If True, print training progress')
	args = parser.parse_args()
	return args


def main():
	# Read the arguments
	args = parse_arg()
	load_model = args.load_model
	verbose = args.verbose
	dataset = args.dataset
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cpu'

	# Read data
	data_dict, unnormalize = CIFAR10(verbose=verbose, device=device)
	X_train = data_dict['X_train']
	y_train = data_dict['y_train']
	X_val = data_dict['X_val']
	y_val = data_dict['y_val']

	data = X_train, y_train, X_val, y_val, dataset
	input_size = X_train.shape[1]
	output_size = torch.nn.functional.one_hot(y_train).shape[1]

	# hyperparameters
	hidden_sizes = [2000,1000,100]
	epochs = 10
	batch_size = 256
	lr = 1e-3


	# Model initialization
	model = feed_forward(input_size, hidden_sizes, output_size)
	prev_epoch = 0
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	criterion = torch.nn.CrossEntropyLoss()
	summary(model, (1,input_size), device='cpu')

	# Move to GPU if available
	model = model.to(device)
	X_train = X_train.to(device)
	y_train = y_train.to(device)
	X_val = X_val.to(device)
	y_val = y_val.to(device)

	# Load previously trained model if specified
	if not load_model == '':
		prev_epoch, _, _, _ = utils.load_checkpoint(model, optimizer, criterion, load_model, verbose)


	# Training
	if verbose:
		print('\nStart training: epoch={}, prev_epoch={}, batch_size={}, lr={}'.format(epochs, prev_epoch, batch_size, lr))
	best_loss, train_acc, val_acc, total_time = fit_ff(model,
		data, 
		optimizer,
		criterion,
		prev_epoch=prev_epoch,
		epochs=epochs,  
		batch_size=batch_size, 
		save_every=1,
		verbose=verbose)

	if verbose:
		print('\nTraining finished! \nTime: {:2f}, best train loss: {:5f}, best train acc: {:5f}, best val acc: {:5f}'.format(total_time, best_loss, train_acc, val_acc))

if __name__ == '__main__':
	main()