# Adversarial Augmentation

## Defense-GAN

It uses Python 2.7, use Anaconda 5.2 for safety when creating a new environment. 

Install using `python2 -m pip install -r requirements.txt`. Tensorflow for GPU seems like 1.14 is working, but not entirely sure (it does not give error but after iteration 4 it stops).

Run the training using `python2 train.py --cfg experiments/cfgs/gans/mnist.yml --is_train`.

## White box attacks investigation

Use transfer learning techniques to slightly modify an already-trained model (such as ResNet) and make it work on a given dataset (such as CIFAR-10). (https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

In a white-box scenario, use the Fast Gradient Sign Method to generate adversarial examples. (https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

The model has been trained and achieves an accuracy of `93%` ca., and is used to obtain various measures.

The FID is used to measure how much our network thinks an image is different from the original when we feed an augmented image. This is useful to evaluate the effects of augmentations on the prediction-capabilities of the model.

We then measure also the calibration of our network to find whether its output correctly predicts the class-belonging probability of the sample. When a low probability is selected the network should make more mistakes, thus showing that it is possible to make correct judgements on the certainty of the prediction based on the class-belonging probability that the network outputs.


## Papers

Defense-GAN:  https://arxiv.org/pdf/1805.06605.pdf

Review of the adversarial literature: https://arxiv.org/pdf/1909.08072.pdf

Theoretical "historical" paper (goodfellow and bengio): https://arxiv.org/pdf/1611.01236.pdf

Carilbration of Neural Networks: https://arxiv.org/pdf/1706.04599.pdf

FID score origin: https://arxiv.org/pdf/1706.08500.pdf
