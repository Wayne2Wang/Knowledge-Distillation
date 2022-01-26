import math
import time
import torch
from torchsummary import summary


import utils
from models import feed_forward


def train_ff(
	data, 
	load_model='', 
	epochs=100, 
	hidden_sizes = [2000,1000,100], 
	batch_size=256, 
	lr=0.001, 
	gpu=True, 
	verbose=False):

	# Read data
	X_train, y_train, X_val, y_val = data
	train_size = X_train.shape[0]
	input_size = X_train.shape[1]
	output_size = torch.nn.functional.one_hot(y_train).shape[1]

	# Model initialization
	if not load_model == '':
		model, prev_epoch, optimizer, criterion, best_loss = utils.load_checkpoint(load_model, verbose)
	else:
		model = feed_forward(input_size, hidden_sizes, output_size)
		prev_epoch = 0
		optimizer = torch.optim.AdamW(model.parameters())
		criterion = torch.nn.CrossEntropyLoss()
		best_loss = float('inf')
	summary(model, (1,input_size))


    # Move to GPU if specified
	if gpu:
		ff_model = model.cuda()
		X_train = X_train.cuda()
		y_train = y_train.cuda()
		X_val = X_val.cuda()
		y_val = y_val.cuda()
  
	# Initialize training
	start_time = time.time()
	if verbose:
		print('Start training: \{}, {}, batch_size={}, epoch={}, lr={}\n'.format(optimizer.__name__, criterion.__name__, batch_size, epochs, lr))
  
    # Start training
	for epoch in range(epochs):

		# Shuffle the data for each epoch
		shuffler = torch.randperm(train_size)
		X_train_shuffled = X_train[shuffler]
		y_train_shuffled = y_train[shuffler]

		total_loss = 0
		new_best_loss = False
		model.train()
		for i in range(math.floor(train_size/batch_size)):

			# Prepare mini-batch
			start = i*batch_size
			if (i+1)*batch_size > train_size:
				end = train_size
			else:
				end = (i+1)*batch_size
				x_batch = X_train_shuffled[start:end]
				y_batch = y_train_shuffled[start:end]

			# Clear gradient
			optimizer.zero_grad()

			# Forward pass
			y_pred = ff_model(x_batch)

			# Compute Loss
			loss = criterion(y_pred, y_batch)

			# Record loss
			loss_value = loss.item()
			total_loss += loss_value
			if loss_value < best_loss:
				best_loss = loss_value
				new_best_loss = True

			# Backward pass
			loss.backward()
			optimizer.step()

		if verbose:
			print('Epoch {}: train loss {}'.format(epoch+1, total_loss/train_size))
		if new_best_loss:
			utils.save_checkpoint('log\\imagenet\\best_loss_model.pt',model, prev_epoch+epoch+1, optimizer, criterion, best_loss)
	
	if verbose:
		print('\nTraining finished! Time: {:2f}, best train loss: {}'.format(time.time()-start_time, best_loss))
	
	return model



def main():
	pass



if __name__ == '__main__':
	main()