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

plt.ion()


class unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ## ----   ARGS   ---- ##
dataset = "cifar_10"
batch_size = 4
num_workers = 4
data_dir = "{}/{}".format("data", dataset)
# ## ---- END ARGS ---- ##

datasets = {
    "cifar_10": {
        "train": datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=data_transforms["train"]
        ),
        "val": datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=data_transforms["val"]
        )
    }
}

dataloaders = {
    "cifar_10": {
        "train": torch.utils.data.DataLoader(datasets["cifar_10"]["train"], batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers),
        "val": torch.utils.data.DataLoader(datasets["cifar_10"]["val"], batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers)

    }
}

dataset_sizes = {dataset_name: {x: len(datasets[dataset_name][x]) for x in ['train', 'val']} for dataset_name in
                 datasets.keys()}
class_names = {dataset_name: datasets[dataset_name]['train'].classes for dataset_name in datasets.keys()}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
unnorm = unnormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

selected_dataset = datasets[dataset]
selected_dataloader = dataloaders[dataset]
selected_class_names = class_names[dataset]
selected_dataset_sizes = dataset_sizes[dataset]


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


inputs, classes = next(iter(selected_dataloader['train']))
print(class_names)
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[selected_class_names[x] for x in classes])

resnet18 = models.resnet18(pretrained=True)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in selected_dataloader[phase]:
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

            epoch_loss = running_loss / selected_dataset_sizes[phase]
            epoch_acc = running_corrects.double() / selected_dataset_sizes[phase]

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


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(selected_dataloader['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}, real label: {}'.format(selected_class_names[preds[j]], selected_class_names[labels[j].item()]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


resnet18 = torchvision.models.resnet18(pretrained=True)
for param in resnet18.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = resnet18.fc.in_features
# modify last layer of ResNet to have the correct number of classes
resnet18.fc = nn.Linear(num_ftrs, len(selected_class_names))

resnet18 = resnet18.to(device)

# loss function
criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(resnet18.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


resnet18 = train_model(resnet18, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=1)

visualize_model(resnet18)

plt.ioff()
plt.show()

epsilons = [0, .05, .1, .15, .2, .25, .3]

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

def test( model, device, test_loader, epsilon ):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for inputs, targets in test_loader:
        # the inputs are batched

        # Send the data and label to the device
        inputs, targets = inputs.to(device), targets.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        inputs.requires_grad = True

        # Forward pass the data through the model
        outputs = model(inputs)
        _, init_preds = torch.max(outputs, 1)  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        for j in range(inputs.size()[0]):
            if init_preds[j].item() != targets[j].item():
                print(init_preds[j])
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
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_preds[j].item(), final_preds[j].item(), adv_ex) )
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_preds[j].item(), final_preds[j].item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(resnet18, device, selected_dataloader["val"], eps)
    accuracies.append(acc)
    examples.append(ex)