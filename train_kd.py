import torch
import tqdm
import torchmetrics

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
