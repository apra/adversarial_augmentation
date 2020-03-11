# Adversarial Augmentation

Defense-GAN

It uses Python 2.7, use Anaconda 5.2 for safety when creating a new environment. 

Install using `python2 -m pip install -r requirements.txt`. Tensorflow for GPU seems like 1.14 is working, but not entirely sure (it does not give error but after iteration 4 it stops).

Run the training using `python2 train.py --cfg experiments/cfgs/gans/mnist.yml --is_train`.
