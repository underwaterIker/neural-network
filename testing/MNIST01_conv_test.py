from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

from network import load_network


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


# load the already trained network
myNetwork = load_network('MNIST01_conv_network.csv')


# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test, y_test = preprocess_data(x_test, y_test, 100)


# test
for x, y in zip(x_test, y_test):
    output = myNetwork.predict(x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
