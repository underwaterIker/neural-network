import numpy as np
from layers import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
        # dE/dx=(dE/dy)⊙f'(x)   |   ⊙ --> element-wise multiplication (np.multiply())

    def save_layer(self, csv_writer):
        csv_writer.writerow([__name__+'.'+type(self).__name__+'()'])

    def load_layer(self, csv_reader):
        pass


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            x[x<=0] = 0
            x[x>0] = 1
            return x

        super().__init__(relu, relu_prime)


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            #print('output before Sigmoid():', x)
            # print(1 / (1 + np.exp(-x)))
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
