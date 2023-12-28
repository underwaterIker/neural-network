import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from layers import Dense, Convolutional, Flatten, Softmax
from activations import Sigmoid, ReLU
from losses import mse, mse_prime, binary_cross_entropy, binary_cross_entropy_prime
from network import Network


def preprocess_data(x, y, limit):
    x = x[:limit]
    y = y[:limit]
    x = x.reshape(len(x), 28, 28, 1)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y


# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = Network()
network.layers = [
    Convolutional(5, 3, initializer='xavier'),
    Sigmoid(),
    Flatten(),
    Dense(100, initializer='xavier'),
    Sigmoid(),
    Dense(10, initializer='xavier'),
    Softmax()
]

# train
network.train(
    mse,
    mse_prime,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=1,
    epochs=20,
    learning_rate=0.1
)
network.save_network('MNIST_conv_network.csv')
