import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import augment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test(model, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data_batch, target_batch in test_loader:
        for data, target in zip(data_batch, target_batch):
            # Send the data and label to the device
            #TODO make this image-size independent, the number of channels can be static
            data = data.view(1, 3, 32, 32)
            target = target.view(1)
            data, target = data.to(device), target.to(device)

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.item() != target.item():
                continue

            # Calculate the loss
            loss = F.nll_loss(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, epsilon, data_grad)

            # Re-classify the perturbed image
            output = model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if final_pred.item() == target.item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader) * test_loader.batch_size)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct,
                                                             len(test_loader) * test_loader.batch_size, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def test_augmented(model, test_loader, epsilon, n = 2, depth = 1, augmentations="all"):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data_batch, target_batch in test_loader:
        #TODO make this work with batches to allow parallelization (requires some bookkeeping for all the performances)
        for data, target in zip(data_batch, target_batch):
            # Send the data and label to the device
            data = data.view(1, 3, 32, 32)
            target = target.view(1)

            # the same target for all of the augmented images
            target, data  = target.to(device), data.to(device)

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.item() != target.item():
                continue

            # Calculate the loss
            loss = F.nll_loss(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, epsilon, data_grad).detach()

            augmented_images, augmented_seq, augmented_seq_name = augment.compute_augmentations(perturbed_data.cpu(), n, depth, augmentations)
            predictions_image = []
            augmented_images = augmented_images
            for aug_img in augmented_images:
                aug_img = aug_img.to(device)
                # Re-classify the perturbed and AUGMENTED image

                output = model(aug_img)

                predictions_image.append(output)

            votes = np.zeros((predictions_image[0].shape[1], 1))

            for prediction in predictions_image:
                # Check for success
                votes[prediction.max(1, keepdim=True)[1].item()]+=1  # get the index of the max log-probability
            final_pred = np.argmax(votes)

            if final_pred == target.item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred, adv_ex))
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred, adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader) * test_loader.batch_size)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct,
                                                             len(test_loader) * test_loader.batch_size, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
