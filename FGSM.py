import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import augment
import math
import torch.distributions as td

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fgsm_attack_batch(batch, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    adversarial_batch = batch + epsilon * sign_data_grad
    # Return the perturbed image
    return adversarial_batch


def test_augmented(model, data_loader, epsilon, mode="mean", n=2, depth=1, augmentations="all", v=0, n_examples=100):
    # Accuracy counter
    correct = 0
    adv_examples = []

    data_loader_len = 0
    test_iteration = 0
    last_seen = 0
    if v > 0:
        data_loader_len = len(data_loader)

    pred_log_probs = []
    real_labels = []

    # Loop over all examples in test set
    for data_batch, target_batch in data_loader:
        if v > 0:
            test_iteration += 1
            percent = round(100 * (test_iteration / data_loader_len))
            if percent > last_seen:
                print_every = 10
                if percent % print_every == 0:
                    print("{}%".format(percent))
                last_seen = percent

        real_labels.append(target_batch.cpu())
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

            pred_log_probs.append(mean_pred_batch.detach().cpu())

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
                    if (epsilon == 0) and (len(adv_examples) < n_examples):
                        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                        adv_examples.append((init_pred.item(), final_pred, adv_ex))
                else:
                    # Save some adv examples for visualization later
                    if len(adv_examples) < n_examples:
                        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                        adv_examples.append((init_pred.item(), final_pred, adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(data_loader) * data_loader.batch_size)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct,
                                                             len(data_loader) * data_loader.batch_size, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, pred_log_probs, real_labels

'''
getExamples creates three lists, one list containing the natural errors. One list containing
correcly classified examples and a final list containing adversarials. Each entry in the list contains the following:
[true class, [the model predicted class, model augmented predicted class], [image, augmented image], [softmax of output of model,softmax of output for augmented image]]
'''
def getExamples(model, data_loader, epsilon = 0.01, n=100, augmentations="flr"):
    n_samples = n-1
    miss_classified_examples = []
    adv_examples = []
    correct_classified_examples = []
    limit1 = 0
    limit2 = 0

    for data_batch, target_batch in data_loader:
        if (limit1 == n_samples) and (limit2 == n_samples): # stop when we got the samples
            break

        # TODO: This loop could probably be done in a smarter way for speed up
        for data, target in zip(data_batch, target_batch):
            data=data.view(1,3,32,32)
            # TODO: make so compute_augmentations takes all inputs, right now only accounts for flip
            augmented_batch, _, _ = augment.compute_augmentations(data.cpu(),
                                    n=1, depth=1, augmentations=augmentations, flip_p=1) 
            
            # concatenate the true image and the augmented image
            data=torch.cat((data, augmented_batch.view(-1,3,32,32)), 0)

            target=torch.cat((target.view(1),target.view(1)), 0)
            data, target = data.to(device), target.to(device)

            data.requires_grad = True
            output =  F.log_softmax(model(data),dim=1)
            init_pred = output.max(1, keepdim=True)[1]

            # If the initial prediction is wrong, dont bother attacking, just move on, 
            # but save the natural error example.
            if init_pred[0].item() != target[0].item():
                if len(miss_classified_examples) <= n_samples:
                    limit1 += 1
                    data_miss=data.detach().cpu()
                    output_miss=output.detach().cpu()
                    miss_classified_examples.append((target.detach().cpu(), init_pred.detach().cpu(), data_miss, output_miss.exp())) 
                continue

            # Call FGSM Attack
            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = fgsm_attack_batch(data, epsilon, data_grad)

            # Re-classify the perturbed image
            output_adv = F.log_softmax(model(perturbed_data),dim=1)

            # Check for successful attack
            final_pred = output_adv.max(1, keepdim=True)[1]
            
            # Save some adversarial examples and correct classified images
            if final_pred[0].item() != target[0].item():
                if len(adv_examples) <= n_samples:
                    limit2 += 1
                    #save correct image
                    data_correct=data.detach().cpu()
                    output_correct=output.detach().cpu()
                    correct_classified_examples.append((target.detach().cpu(), 
                                                        init_pred.detach().cpu(), 
                                                        data_correct, output_correct.exp())) 
                    
                    #save adversarial
                    out_adv = output_adv.detach().cpu()
                    adv_ex = perturbed_data.detach().cpu()
                    adv_examples.append( (target.detach().cpu(), final_pred.detach().cpu(), adv_ex, out_adv.exp()))
                
    return correct_classified_examples, miss_classified_examples, adv_examples

'''
Helper function to get data from getAugmentImages
'''
def getOutput(data):
    target = []
    pred = []
    image = []
    out = []
    for i,j,k,l in data:
        target.append(i)
        pred.append(j.view(-1))
        image.append(k)
        out.append(l)
    return target, pred, image, out

'''
Compute the KLD of the output||augmented_output
'''        
def DKL(out_softmax):
    dkl = []
    for data in out_softmax:
        p=td.Categorical(probs=data[0])
        q=td.Categorical(probs=data[1])
        dkl.append(td.kl_divergence(p, q))
    return dkl

'''
Calculate data for plots
'''
def DKLBin(dkl, bin_size = 0.2, bin_max=10):
    dkl_count = []
    bins_start = np.arange(0, bin_max, bin_size)
    bins_end = bins_start+bin_size
    bin_length = len(bins_start) 
    n_images = len(dkl)
    for i in range(bin_length):
        counter = 0
        for j in range(n_images):
            if (bins_start[i] <= dkl[j]) and (bins_end[i] > dkl[j]):
                counter+=1
        dkl_count.append(counter/n_images)
    return dkl_count, bins_start