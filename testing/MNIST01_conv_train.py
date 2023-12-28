import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from layers import Dense, Convolutional, Flatten, MaxPooling, Dropout, Softmax
from activations import Sigmoid, ReLU
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import Network


def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 28, 28, 1)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y


# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = Network()
network.layers = [
    Convolutional(num_filters=5, kernel_size=3, initializer='he'),
    ReLU(), # ReLU gives better results?
    MaxPooling(2, 2),
    Dropout(0.8),
    Flatten(),
    Dense(100, initializer='he'),
    ReLU(),
    Dense(2, initializer='xavier'),
    Softmax()
]

# train
network.train(
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=1,
    epochs=1,
    learning_rate=0.001)
network.save_network('MNIST01_conv_network.csv')
