import torch
import torchvision
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import augment
import math
import torch.distributions as td
import FGSM

# plotting errorbars on histograms
import inspect
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_good_adversarial(model, dataloader, epsilon, N, n_classes=10, adversarial_attack=FGSM.fgsm_attack_batch):
    samples_class = {}
    for i in range(n_classes):
        samples_class[i] = []
    for data_batch, target_batch in dataloader:
        finished_finding_samples = True
        for i in range(n_classes):
            if len(samples_class[i]) < N:
                finished_finding_samples = False
        if finished_finding_samples:
            break
        for original_data, original_target in zip(data_batch, target_batch):

            data = original_data.view(1, 3, 32, 32).to(device)
            target = original_target.view(1).to(device)

            data.requires_grad = True
            output = F.log_softmax(model(data), dim=1)
            init_pred = output.max(1, keepdim=True)[1]

            if init_pred[0].item() != target[0].item():
                continue
            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            # create adversarial
            perturbed_data = adversarial_attack(data, epsilon, data_grad)

            # run new classification using adversarial attack
            output_adv = F.log_softmax(model(perturbed_data), dim=1)
            final_pred = output_adv.max(1, keepdim=True)[1]

            if final_pred[0].item() != target[0].item():
                if len(samples_class[original_target.item()]) < N:
                    # each element is ((input data, original class), (adversarial data, predicted class))
                    samples_class[original_target.item()].append(((original_data, original_target), (
                        perturbed_data.squeeze().detach().cpu(), final_pred.squeeze().detach().cpu())))

    return samples_class


def get_samples_from_classes(model, dataloader, N, n_classes=10, missclass=False):
    samples_class = {}
    for i in range(n_classes):
        samples_class[i] = []
    for data_batch, target_batch in dataloader:
        finished_finding_samples = True
        for i in range(n_classes):
            if len(samples_class[i]) < N:
                finished_finding_samples = False
        if finished_finding_samples:
            break
        for original_data, original_target in zip(data_batch, target_batch):

            data = original_data.view(1, 3, 32, 32).to(device)
            target = original_target.view(1).to(device)

            output = F.log_softmax(model(data), dim=1)
            pred = output.max(1, keepdim=True)[1]

            if missclass:
                criterion = pred[0].item() != target[0].item()
            else:
                criterion = pred[0].item() == target[0].item()

            if criterion:
                if len(samples_class[original_target.item()]) < N:
                    # each element is: (input data, (original class, predicted class))
                    samples_class[original_target.item()].append(
                        (original_data, (original_target, pred.squeeze().detach().cpu())))

    return samples_class


import torch.distributions as td


def get_kl_augmentations(model, datalist, augmentations, n):
    dkl = []
    for data in datalist:
        data = data.view(1, 3, 32, 32).to(device)
        # only one augmentation at a time
        assert len(augmentations.split(",")) >= 1
        augmented_batch, _, _ = augment.compute_augmentations(data.detach().cpu(),
                                                              n=n, depth=1, augmentations=augmentations, flip_p=1,
                                                              rot=(-15, -14.9))

        for i in range(n):
            data = torch.cat((data, augmented_batch[i].to(device)), 0)
        # concatenate the true image and the augmented image
        output = F.log_softmax(model(data), dim=1)

        init_pred = output[0].exp()
        aug_pred = torch.mean(output[1:].exp(), dim=0)

        p = td.Categorical(probs=init_pred.detach().cpu())
        q = td.Categorical(probs=aug_pred.detach().cpu())
        dkl.append(td.kl_divergence(p, q).item())

    return dkl


def get_l1_norm__augmentations(model, datalist, augmentations, n):
    l1_norms = []
    for data in datalist:
        data = data.view(1, 3, 32, 32).to(device)
        # only one augmentation at a time
        assert len(augmentations.split(",")) >= 1
        augmented_batch, _, _ = augment.compute_augmentations(data.detach().cpu(),
                                                              n=n, depth=1, augmentations=augmentations, flip_p=1,
                                                              rot=(-15, -14.9))

        for i in range(n):
            data = torch.cat((data, augmented_batch[i].to(device)), 0)
        # concatenate the true image and the augmented image
        output = F.log_softmax(model(data), dim=1)

        init_pred = output[0].exp()
        aug_pred = torch.mean(output[1:].exp(), dim=0)

        l1_norms.append(torch.norm(init_pred, aug_pred, p=1).item())

    return l1_norms



'''
Plot error-bars on histograms
'''


def hist_errorbars(data, xerrs=True, label="", color="r", *args, **kwargs):
    """Plot a histogram with error bars. Accepts any kwarg accepted by either numpy.histogram or pyplot.errorbar"""
    # pop off normed kwarg, since we want to handle it specially
    norm = False
    if 'normed' in kwargs.keys():
        norm = kwargs.pop('normed')

    # retrieve the kwargs for numpy.histogram
    histkwargs = {}
    for key, value in kwargs.items():
        if key in inspect.signature(np.histogram).parameters.keys():
            histkwargs[key] = value

    histvals, binedges = np.histogram(data, **histkwargs)
    yerrs = np.sqrt(histvals)

    if norm:
        nevents = float(sum(histvals))
        binwidth = (binedges[1] - binedges[0])
        histvals = histvals / nevents / binwidth
        yerrs = yerrs / nevents / binwidth

    bincenters = (binedges[1:] + binedges[:-1]) / 2

    if xerrs:
        xerrs = (binedges[1] - binedges[0]) / 2
    else:
        xerrs = None

    # retrieve the kwargs for errorbar
    ebkwargs = {}
    for key, value in kwargs.items():
        if key in inspect.signature(plt.errorbar).parameters.keys():
            ebkwargs[key] = value
    out = plt.errorbar(bincenters, histvals, yerrs, xerrs, fmt="s-", capsize=3, label=label, color=color, **ebkwargs)

    if 'log' in kwargs.keys():
        if kwargs['log']:
            plt.yscale('log')

    if 'range' in kwargs.keys():
        plt.xlim(*kwargs['range'])

    return out


