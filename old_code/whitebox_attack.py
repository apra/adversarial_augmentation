import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
from dataset_loading import selected_dataloader, selected_class_names, selected_dataset, selected_dataset_sizes, device

# ## ----   ARGS   ---- ##
dataset_name = "cifar10"
batch_size = 4
num_workers = 4
data_dir = "{}/{}".format("data", dataset_name)
model_name = "resnet18"
# ## ---- END ARGS ---- ##

selected_model = models.resnet18()
num_ftrs = selected_model.fc.in_features
selected_model.fc = nn.Linear(num_ftrs, len(selected_class_names))
selected_model.load_state_dict(torch.load("models/{}_{}_model".format(model_name, dataset_name), map_location=device))

epsilons = [.1, .15, .2, .25, .3]
def imshow(inp, title=None):
    """Imshow for Tensor."""

    inp = inp.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test( model, device, test_loader, epsilon, criterion ):
    # Accuracy counter
    correct = 0
    adv_examples = []
    k = 0
    # Loop over all examples in test set
    for inputs, targets in test_loader:
        # the inputs are batched
        print(k)
        k+=1
        # Send the data and label to the device
        inputs, targets = inputs.to(device), targets.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        inputs.requires_grad = True

        # Forward pass the data through the model
        outputs = model(inputs)
        _, init_preds = torch.max(outputs, 1)  # get the index of the max log-probability
        wrong_prediction_flag = False
        # If the initial prediction is wrong, dont bother attacking, just move on
        for j in range(inputs.size()[0]):
            if init_preds[j].item() != targets[j].item():
                wrong_prediction_flag = True

        if wrong_prediction_flag:
            continue
        # Calculate the loss
        loss = criterion(outputs, targets)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = inputs.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(inputs, epsilon, data_grad)

        # Re-classify the perturbed image
        outputs = model(perturbed_data)

        # Check for success
        _,final_preds = torch.max(outputs, 1)  # get the index of the max log-probability
        for j in range(inputs.size()[0]):
            if final_preds[j].item() == targets[j].item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data[j].squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_preds[j].item(), final_preds[j].item(), adv_ex) )
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data[j].squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_preds[j].item(), final_preds[j].item(), adv_ex) )
                    print("New adversary example found")
                    imshow(adv_ex, "Adversary example, labelled: {}".format(selected_class_names[final_preds[j]]))
                    imshow(inputs[j].detach().cpu().numpy(), "Original image, labelled: {}".format(selected_class_names[init_preds[j]]))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

criterion = nn.CrossEntropyLoss()

# Run test for each epsilon
for eps in epsilons:
    print(eps)
    acc, ex = test(selected_model, device, selected_dataloader["val"], eps, criterion)
    accuracies.append(acc)
    examples.append(ex)