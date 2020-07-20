import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#VISUALIZE
def imshow(inputs, title=None, savefigure="temp"):
    """Imshow for Tensor."""
    inp = torchvision.utils.make_grid(inputs)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # pause a bit so that plots are updated
    plt.savefig("figures/"+savefigure)
    plt.pause(0.001)
def prep_imshow (inputs):
    inp = torchvision.utils.make_grid(inputs)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    inp = std * inp + mean
    return np.clip(inp, 0, 1)

def display_images(inputs, classes, figurename=None):
    inp = prep_imshow(inputs)
    print(inp.shape)
    plt.imshow(inp)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',
        left=False,  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,
        labelleft=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    for i, tit in enumerate(classes):
        plt.text(i * (inp.shape[0]-2) + inp.shape[0]/2, -5, tit, fontsize=12, horizontalalignment='center')
    if figurename is not None:
        plt.savefig("figures/" + figurename)
    plt.pause(0.001)

    
def visualize_model(model, testloader, classes, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(classes[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)