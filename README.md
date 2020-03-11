# Adversarial Augmentation

Defense-GAN

It uses Python 2.7, use Anaconda 5.2 for safety when creating a new environment. 

Install using `python2 -m pip install -r requirements.txt`. Tensorflow for GPU seems like 1.14 is working, but not entirely sure (it does not give error but after iteration 4 it stops).

Run the training using `python2 train.py --cfg experiments/cfgs/gans/mnist.yml --is_train`.


## Papers

Defense-GAN:  https://arxiv.org/pdf/1805.06605.pdf

Review of the adversarial literature: https://arxiv.org/pdf/1909.08072.pdf

Theoretical "historical" paper (goodfellow and bengio): https://arxiv.org/pdf/1611.01236.pdf