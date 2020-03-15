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