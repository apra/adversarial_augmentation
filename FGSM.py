import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import augment
import math

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


def fgsm_attack_batch(batch, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    adversarial_batch = batch + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return adversarial_batch


def test(model, test_loader, epsilon, v=0):
    # Accuracy counter
    correct = 0
    adv_examples = []

    test_loader_len = 0
    test_iteration = 0
    last_seen = 0
    if v > 0:
        test_loader_len = len(test_loader)

    # Loop over all examples in test set
    for data_batch, target_batch in test_loader:
        if v > 0:
            test_iteration += 1
            percent = round(100 * (test_iteration / test_loader_len))
            if percent > last_seen:
                print_every = 2
                if percent % print_every == 0:
                    print("{}%".format(percent))
                last_seen = percent
        for data, target in zip(data_batch, target_batch):
            # Send the data and label to the device
            data = data.view(1, 3, 32, 32)
            target = target.view(1)
            data, target = data.to(device), target.to(device)

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = F.log_softmax(model(data), dim=1)
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


def test_augmented(model, data_loader, epsilon, mode="mean", n=2, depth=1, augmentations="all", v=0):
    # Accuracy counter
    correct = 0
    adv_examples = []

    test_loader_len = 0
    test_iteration = 0
    last_seen = 0
    if v > 0:
        test_loader_len = len(data_loader)

    predicted_probabilities = []
    real_labels = []

    # Loop over all examples in test set
    for data_batch, target_batch in data_loader:
        if v > 0:
            test_iteration += 1
            percent = round(100 * (test_iteration / test_loader_len))
            if percent > last_seen:
                print_every = 25
                if percent % print_every == 0:
                    print("{}%".format(percent))
                last_seen = percent

        real_labels.append(target_batch)
        # data_batch has dimensions: [batch_size, channels, width, height]
        # target_batch has dimensions: [batch_size], the class is a number between 0 and 9
        # send data and target to the device
        target_batch, data_batch = target_batch.to(device), data_batch.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data_batch.requires_grad = True

        # Forward pass the data through the model, compute log_softmax to get log-probabilities
        log_probs = F.log_softmax(model(data_batch), dim=1)
        init_preds = log_probs.max(1, keepdim=True)[1]  # get the index of the max log-probability

        loss = F.nll_loss(log_probs, target_batch)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        data_grad = data_batch.grad.data

        # Call FGSM Attack and attach to other images
        adversarial_images = fgsm_attack_batch(data_batch, epsilon, data_grad)

        with torch.no_grad():
            # augment the adversarial images, each batch is augmented.
            # shape: [n_augments, batch_size, channels, w, h]. Tt's a tensor in the cpu [n_aug, batch_size, ...]
            augmented_batch, augmented_seq, augmented_seq_name = augment.compute_augmentations(adversarial_images.cpu(),
                                                                                               n, depth, augmentations)

            # augmented_batch size: [n augmentations, batch size, n channels, width, height]
            # [n augmentations, batch size, n classes]
            log_probs_aug_batch = torch.Tensor().to(device)
            for augmentated_images in augmented_batch:
                augmentated_images = augmentated_images.to(device)

                output = F.log_softmax(model(augmentated_images), dim=1)

                log_probs_aug_batch = torch.cat((log_probs_aug_batch, output.unsqueeze(0)), 0)

            mean_pred_batch = torch.mean(log_probs_aug_batch, dim=0)
            std_pred_batch = torch.std(log_probs_aug_batch, dim=0)

            predicted_probabilities.append(mean_pred_batch.detach().cpu())

            if mode == "mean":
                selected_class_batch = mean_pred_batch.max(1, keepdim=True)[1]
            elif mode == "vote":
                num_classes = log_probs_aug_batch.shape[2]
                # I write here what I do because I will forget in 10 minutes:
                #   - output.max(2, keepdim=True)[1] gets the class number for each augmented batch
                #   - F.one_hot(..., num_classes=num_classes) converts the number in a one-hot vector
                #       so [3, 1] -> [[0, 0, 0, 1, 0, ...], [0, 1, 0, 0, 0, ...]]
                #   - torch.sum(..., , dim=0) sums the votes ACROSS augmented batches, so now each batch has the votes
                #   - torch.sum(...).max(2, keepdim=True)[1] picks the most voted class
                #   - the final .squeeze(1) is to remove unwanted dimensions

                selected_class_batch = \
                    torch.sum(F.one_hot(log_probs_aug_batch.max(2, keepdim=True)[1], num_classes=num_classes), dim=0) \
                        .max(2, keepdim=True)[1] \
                        .squeeze(1)
            else:
                selected_class_batch = mean_pred_batch.max(1, keepdim=True)[1]

            for i, target in enumerate(target_batch):
                final_pred = selected_class_batch[i].item()
                perturbed_data = adversarial_images[i]
                init_pred = init_preds[i]
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
    final_acc = correct / float(len(data_loader) * data_loader.batch_size)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct,
                                                             len(data_loader) * data_loader.batch_size, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, predicted_probabilities, real_labels


def test_augmented_statistics(model, test_loader, epsilon, n=2, depth=1, augmentations="all"):
    # Accuracy counter
    correct = 0
    adv_examples = []
    mean_preds = []
    std_preds = []
    k = 1
    tot = len(test_loader)
    # Loop over all examples in test set
    for data_batch, target_batch in test_loader:
        if k % 100 == 0:
            print(k / tot)
        k += 1
        for data, target in zip(data_batch, target_batch):
            # Send the data and label to the device
            data = data.view(1, 3, 32, 32)
            target = target.view(1)

            # the same target for all of the augmented images
            target, data = target.to(device), data.to(device)

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

            augmented_images, augmented_seq, augmented_seq_name = augment.compute_augmentations(perturbed_data.cpu(), n,
                                                                                                depth, augmentations)
            predictions_image = torch.Tensor().to(device)
            for aug_img in augmented_images:
                aug_img = aug_img.to(device)
                # Re-classify the perturbed and AUGMENTED image
                output = model(aug_img)
                predictions_image = torch.cat((predictions_image, output), 0)

            mean_pred = torch.mean(predictions_image, dim=0)
            std_pred = torch.std(predictions_image, dim=0)

            mean_preds.append(mean_pred.detach().cpu())
            std_preds.append(std_pred.detach().cpu())

            # for prediction in predictions_image:
            # Check for success
            #    votes[prediction.max(1, keepdim=True)[1].item()] += 1  # get the index of the max log-probability
            # final_pred = np.argmax(votes)

            final_pred = mean_pred.max(0, keepdim=True)[1]

            if final_pred == target.item():
                correct += 1
                # Special case predictions_imagefor saving 0 epsilon examples
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
    return final_acc, adv_examples, mean_preds, std_preds
