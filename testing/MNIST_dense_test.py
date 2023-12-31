import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from network import load_network


def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[200:200+limit], y[200:200+limit]


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test, y_test = preprocess_data(x_test, y_test, 10)

# load the already trained network
myNetwork = load_network('MNIST_dense_network.csv')

# test
for x, y in zip(x_test, y_test):
    output = myNetwork.predict(x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
