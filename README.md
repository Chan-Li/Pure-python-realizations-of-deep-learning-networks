# Pure-python-realizations-of-deep-learning-networks
In this repository, I will display some realizations of deep learning frameworks: MLP, CNN, RNN, hopfield with pure python, which may offer an advantage for the readers to better understand the mechanisms hidden in the learning process (compared with other frameworks like pytorch.)
## Commonly-used datasets
In order to keep the relative consistence among all the network architectures, I introduce three commonly-used datasets: MNIST dataset, Fashion-MNIST dataset and CIFAR-10 dataset, which will be utilized in training the following networks. Typically, these datasets are used to implement classification tasks.

[MNIST](http://yann.lecun.com/exdb/mnist/) dataset, four ingredents are included: 60000 pictures and corresponding labels in training set, 10000 pictures and corresponding labels in testing set. The range of pixels is often scaled between (0,1). Please first download this dataset, and then load this using [load_MNIST](https://github.com/Chan-Li/Pure-python-realizations-of-deep-learning-networks/blob/main/Datasets/load_MNIST). 
```python
from Datasets import load_MNIST as load2
mnist=(load2.load_mnist(one_hot=True))
train_data = mnist[0][0][0:60000].T
train_label = mnist[0][1][0:60000].T
test_data = mnist[1][0][0:10000].T
test_label = mnist[1][1][0:10000].T
print(np.shape(train_data))
print(np.shape(train_label))
```

[CIFAR 10 and CIFAR 100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html) contains natural images which are much more difficult to classify, including bird, cat, deer... and so on. In this dataset, there are 50000 training pictures along with training labels, and 10000 test pictures with corresponding labels.



There are also other datasets, like [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist). But for simplicity, we use here only the MNIST and CIFAR-10 datasets.
