import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from layers import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import Network


def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 10000)
x_test, y_test = preprocess_data(x_test, y_test, 1000)

# neural network
network = Network()
network.layers = [
    Dense(80, initializer='xavier'),
    Tanh(),
    Dense(10, initializer='xavier'),
    Tanh()
]

# train
network.train(mse, mse_prime, x_train, y_train, x_test, y_test, batch_size=1, epochs=100, learning_rate=0.1)
network.save_network('MNIST_dense_network.csv')
