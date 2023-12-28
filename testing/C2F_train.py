import numpy as np

from layers import Dense
from losses import mse, mse_prime
from network import Network

X = np.reshape([[-87], [-40], [-33], [-28], [-12], [10], [23], [46], [98], [56]], (10, 1, 1))
Y = np.reshape([[-124.6], [-40], [-27.4], [-18.4], [10.4], [50], [73.4], [114.8], [208.4], [132.8]], (10, 1, 1))

X_validation = np.reshape([[-100], [-50], [-30], [-20], [-10], [10], [25], [40], [96], [53]], (10, 1, 1))
Y_validation = np.reshape([[-148], [-58], [-22], [-4], [14], [50], [77], [104], [204.8], [127.4]], (10, 1, 1))

# define network
network = Network()
network.layers = [
    Dense(1, initializer='xavier')
]

# train
network.train(mse, mse_prime, X, Y, X_validation, Y_validation, batch_size=1, epochs=5000, learning_rate=0.0001)
network.save_network('C2F_network.csv')
