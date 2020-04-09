# Adversarial Augmentation

## White box attacks investigation

Use transfer learning techniques to slightly modify an already-trained model (such as ResNet) and make it work on a given dataset (such as CIFAR-10). (https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

In a white-box scenario, use the Fast Gradient Sign Method to generate adversarial examples. (https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

The model has been trained and achieves an accuracy of `93%` ca., and is used to obtain various measures.

The FID is used to measure how much our network thinks an image is different from the original when we feed an augmented image. This is useful to evaluate the effects of augmentations on the prediction-capabilities of the model.

We then measure also the calibration of our network to find whether its output correctly predicts the class-belonging probability of the sample $P(G=k|X=x)$. When a low probability is selected the network should make more mistakes, thus showing that it is possible to make correct judgements on the certainty of the prediction based on the class-belonging probability that the network outputs.


## Papers

Defense-GAN:  https://arxiv.org/pdf/1805.06605.pdf

Review of the adversarial literature: https://arxiv.org/pdf/1909.08072.pdf

Theoretical "historical" paper (goodfellow and bengio): https://arxiv.org/pdf/1611.01236.pdf

Carilbration of Neural Networks: https://arxiv.org/pdf/1706.04599.pdf

FID score original: https://arxiv.org/pdf/1706.08500.pdf

Framework for attacks/defense: https://arxiv.org/pdf/1712.07107.pdf

Some theory for TTA: https://www.sciencedirect.com/science/article/pii/S0925231219301961?via%3Dihub, http://openaccess.thecvf.com/content_cvpr_2018/papers/Radosavovic_Data_Distillation_Towards_CVPR_2018_paper.pdf, https://arxiv.org/pdf/1412.4864.pdf
