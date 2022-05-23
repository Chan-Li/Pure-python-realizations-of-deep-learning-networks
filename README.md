# Pure-python-realizations-of-deep-learning-networks
In this repository, I will display some realizations of deep learning frameworks: MLP, CNN, RNN, hopfield with pure python, which may offer an advantage for the readers to better understand the mechanisms hidden in the learning process (compared with other frameworks like pytorch.)
## Commonly-used datasets
In order to keep the relative consistence among all the network architectures, I introduce three commonly-used datasets: MNIST dataset, Fashion-MNIST dataset and CIFAR-10 dataset, which will be utilized in training the following networks. Typically, these datasets are used to implement classification tasks.

[MNIST](http://yann.lecun.com/exdb/mnist/) dataset, four ingredents are included: 60000 pictures and corresponding labels in training set, 10000 pictures and corresponding labels in testing set. The range of pixels is often scaled between (0,1). Please first download this dataset, and then load this using [load_MNIST](https://github.com/Chan-Li/Pure-python-realizations-of-deep-learning-networks/blob/main/Datasets/load_MNIST).
