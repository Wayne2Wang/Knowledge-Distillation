# Knowledge-Distillation
To train the models on ImageNet: (1) download ImageNet1K 2012; (2) copy val_labels.txt and train_labels.txt to the ImageSet folder


To set up the right environment, run the following command

pip install -r requirements.txt

To train the MLP, run

python train.py

To train a ResNet18(pretrained) for testing purpose, run

python train.py --resnet

To resume training, run

python train.py --load_model PATH_TO_MODEL

To evaluate the pretrained ResNet18, run

python eval.py

To view training stats, run

tensorboard --logdir log/DATASET
