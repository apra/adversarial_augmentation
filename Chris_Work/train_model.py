import torch
import torchvision
import numpy as np
import time
import copy
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.optim import lr_scheduler

def train(model, trainloader, testloader, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()
    #define criterion
    criterion = nn.CrossEntropyLoss()
    #define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    #LR decay, 0.1, every 7 epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    N_train = len(trainloader)*trainloader.batch_size
    N_test = len(testloader)*trainloader.batch_size
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = trainloader
                N=N_train
            else:
                model.eval()   # Set model to evaluate mode
                dataloader=testloader
                N=N_test
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / N
            epoch_acc = running_corrects.double() / N

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model