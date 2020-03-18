import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(batch_size=4):
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    transform_validation = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_validation)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    classes = trainset.classes
    N_tr = len(trainset)
    N_tst=len(testset)
    
    attributes={"class_names": classes, "N_train":N_tr, "N_test":N_tst}
    
    return trainloader, testloader, attributes