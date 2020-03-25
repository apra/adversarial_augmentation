##############
# # IMAGE AUGMENTATION (created: 25/03/2020)
##############
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random

# image augmentation library
import imgaug as ia
from imgaug import augmenters as iaa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
ia.seed(42)


def compute_augmentations(original_tensor, n=1, depth=1, augmentations="all"):
    # list of possible augmentations and parameters
    rotate = iaa.Affine(rotate=(-12, 12))
    gaussian_noise = iaa.AdditiveGaussianNoise(scale=(0, 25))
    # map them to names that can be used
    augmenters = {
        "rotation": rotate,
        "r": rotate,
        "gaussian_noise": gaussian_noise,
        "g": gaussian_noise
    }

    # Correct dimensions of the image, normalize
    original_batch = original_tensor.numpy().transpose((0, 2, 3, 1))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    original_batch = std * original_batch + mean
    original_batch = np.clip(original_batch, 0, 1)
    # convert to 0-255 uint8 format for imgaug
    original_batch = original_batch * 255
    original_batch = original_batch.astype("uint8")

    # get the list of augmentation possible
    augmentation_set = []
    if augmentations == "all":
        augmentation_set = ["rotation", "gaussian_noise"]
    else:
        for aug in augmentations.split(","):
            if aug not in augmenters.keys():
                print("Augmentation: {} is not recognized.\n"
                      "Choose one out of the following:\n{}".format(aug, augmenters.keys()))
            else:
                augmentation_set.append(aug)
    assert len(augmentation_set) > 0
    # prepare for output
    augmented_batches = []

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
        augmented_batches.append(aug_batch_torch)

    return augmented_batches, augmentation_sequences, augmentation_sequences_names
