import numpy as np

from layers import Dense
from activations import ReLU, Tanh, Sigmoid
from losses import mse, mse_prime
from network import Network

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

X_validation = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y_validation = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = Network()
network.layers = [
    Dense(5, initializer='xavier'),
    Tanh(),
    Dense(1, initializer='xavier'),
    Tanh()
]

# train
network.train(mse, mse_prime, X, Y, X_validation, Y_validation, batch_size=1, epochs=2000, learning_rate=0.1)
network.save_network('XOR_network.csv')
