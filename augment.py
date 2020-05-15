##############
# IMAGE AUGMENTATION (created: 25/03/2020)
##############
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random

# image augmentation library
import imgaug as ia
from imgaug import augmenters as iaa

random.seed(42)
ia.seed(42)


def compute_augmentations(original_tensor, n=1, depth=1, augmentations="all",
                          rot=(-12, 12),
                          noise=(0, 25),
                          flip_p=0.5,
                          br_add=100,
                          shearx_amnt= 20,
                          crop_per=0.2
                          ):
    if augmentations == "none":
        return original_tensor.unsqueeze(0), None, None
    # list of possible augmentations and parameters
    rotate = iaa.Affine(rotate=rot, mode="edge")
    gaussian_noise = iaa.AdditiveGaussianNoise(scale=noise)
    flip = iaa.Fliplr(flip_p)
    brightness_add = iaa.WithBrightnessChannels(iaa.Add((-br_add, br_add)))
    shear_x = iaa.ShearX((-shearx_amnt, shearx_amnt))
    crop_per = min(0.999,max(crop_per,0.001))
    cropandpad = iaa.CropAndPad(percent=(-crop_per, crop_per), pad_mode="edge")
    # map them to names that can be used
    augmenters = {
        "rotation": rotate,
        "r": rotate,
        "gaussian_noise": gaussian_noise,
        "g": gaussian_noise,
        "fliplr": flip,
        "flr": flip,
        "brightness_add": brightness_add,
        "bra": brightness_add,
        "shear_x": shear_x,
        "cropandpad": cropandpad
    }

    # Correct dimensions of the image, normalize
    original_batch = original_tensor.numpy().transpose((0, 2, 3, 1))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    original_batch = std * original_batch + mean
    #make sure stuff is between 0 and 1
    original_batch = np.clip(original_batch, 0, 1)
    # convert to 0-255 uint8 format for imgaug
    original_batch = original_batch * 255
    original_batch = original_batch.astype("uint8")

    # get the list of augmentation possible
    augmentation_set = []
    if augmentations == "all":
        augmentation_set = ["rotation", "gaussian_noise", "fliplr", "brightness_add"]
    else:
        for aug in augmentations.split(","):
            aug = aug.strip()
            if aug not in augmenters.keys():
                print("Augmentation: {} is not recognized.\n"
                      "Choose one out of the following:\n{}".format(aug, augmenters.keys()))
            else:
                augmentation_set.append(aug)
    assert len(augmentation_set) > 0
    # prepare for output (keep it in the cpu)
    augmented_batches = torch.Tensor()

    # the sequence of augmentations to apply to the batch of images
    augmentation_sequences = []
    augmentation_sequences_names = []

    for i in range(n):
        augmentation_sequences.append([])
        augmentation_sequences_names.append([])
        for _ in range(depth):
            augmentation = random.choice(augmentation_set)
            augmentation_sequences_names[i].append(augmentation)
            augmentation_sequences[i].append(augmenters[augmentation])
        seq = iaa.Sequential(augmentation_sequences[i])
        # change type
        aug_batch_numpy = seq(images=original_batch).astype("float") / 255.
        # normalize
        aug_batch_numpy = (aug_batch_numpy - mean) / std
        # bring to pytorch
        aug_batch_torch = torch.from_numpy(aug_batch_numpy.transpose((0, 3, 1, 2))).float()
        augmented_batches = torch.cat((augmented_batches, aug_batch_torch.unsqueeze(0)), 0)

    return augmented_batches, augmentation_sequences, augmentation_sequences_names
