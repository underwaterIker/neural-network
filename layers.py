import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass

    def save_layer(self, csv_writer):
        # TODO: save all the layer parameters in a .csv
        pass

    def load_layer(self, csv_reader):
        # TODO: load the layer from a .csv where all the layer parameters are stored
        pass


class Dense(Layer):
    def __init__(self, output_size, initializer):
        self.output_size = output_size
        self.initializer = initializer
        self.weights = None
        self.bias = np.zeros(shape=(output_size, 1))

    def xavier_initializer(self, output_size, input_size): # para activaciones Tanh o Sigmoid
        variance = 2.0 / (input_size + output_size)
        stddev = np.sqrt(variance)
        return np.random.randn(output_size, input_size) * stddev

    def he_initializer(self, output_size, input_size): # para activacion ReLU
        variance = 2.0 / input_size
        stddev = np.sqrt(variance)
        return np.random.randn(output_size, input_size) * stddev

    def forward(self, input):
        self.input = input

        if self.weights is None: # first time
            if self.initializer == 'xavier':
                self.weights = self.xavier_initializer(self.output_size, self.input.shape[0])
            elif self.initializer == 'he':
                self.weights = self.he_initializer(self.output_size, self.input.shape[0])
            else:
                print('Initializer not defined properly. --> (Dense layer)')

        output = np.dot(self.weights, self.input) + self.bias
        return output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

    def save_layer(self, csv_writer):
        first_row = [__name__+'.'+type(self).__name__+'('+str(self.output_size)+', "'+str(self.initializer)+'")']
        csv_writer.writerow(first_row)
        csv_writer.writerows(self.weights)
        csv_writer.writerows(self.bias)

    def load_layer(self, csv_reader):
        self.weights = []
        for j in range(self.output_size):
            self.weights.append([float(k) for k in next(csv_reader)])
        for l in range(self.output_size):
            self.bias[l] = [float(m) for m in next(csv_reader)]


class Convolutional(Layer):
    def __init__(self, num_filters, kernel_size, initializer, padding=0, stride=1):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.initializer = initializer

        self.weights = None
        self.bias = np.zeros(num_filters)

    def xavier_initializer(self, input_channels, num_filters): # para activaciones Tanh o Sigmoid
        variance = 2.0 / (input_channels + num_filters)
        stddev = np.sqrt(variance)
        return np.random.randn(self.kernel_size, self.kernel_size, input_channels, num_filters) * stddev

    def he_initializer(self, input_channels): # para activacion ReLU
        variance = 2.0 / input_channels
        stddev = np.sqrt(variance)
        return np.random.randn(self.kernel_size, self.kernel_size, input_channels, self.num_filters) * stddev

    def forward(self, input):
        self.input = input
        input_height, input_width, input_channels = input.shape
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = np.zeros((output_height, output_width, self.num_filters))

        if self.weights is None: # first time
            if self.initializer == 'xavier':
                self.weights = self.xavier_initializer(input_channels, self.num_filters)
            elif self.initializer == 'he':
                self.weights = self.he_initializer(input_channels)
            else:
                print('Initializer not defined properly. --> (Convolutional layer)')

        for i in range(output_height):
            for j in range(output_width):
                for f in range(self.num_filters):
                    i_start = i * self.stride - self.padding
                    i_end = i_start + self.kernel_size
                    j_start = j * self.stride - self.padding
                    j_end = j_start + self.kernel_size
                    receptive_field = input[i_start:i_end, j_start:j_end, :]
                    output[i, j, f] = np.sum(receptive_field * self.weights[:, :, :, f]) + self.bias[f]

        return output

    def backward(self, grad_output, learning_rate):
        output_height, output_width, num_filters = grad_output.shape
        grad_input = np.zeros_like(self.input)
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.zeros(num_filters)

        for i in range(output_height):
            for j in range(output_width):
                for f in range(num_filters):
                    i_start = i * self.stride - self.padding
                    i_end = i_start + self.kernel_size
                    j_start = j * self.stride - self.padding
                    j_end = j_start + self.kernel_size

                    grad_input[i_start:i_end, j_start:j_end, :] += grad_output[i, j, f] * self.weights[:, :, :, f]
                    receptive_field = self.input[i_start:i_end, j_start:j_end, :]
                    grad_weights[:, :, :, f] += grad_output[i, j, f] * receptive_field
                    grad_bias[f] += grad_output[i, j, f]

        # Actualizar pesos y sesgos
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input



    # def __init__(self, input_shape, num_filters, kernel_size):
    #     input_height, input_width, input_channels = input_shape
    #     self.num_filters = num_filters
    #     self.input_shape = input_shape
    #     self.num_channels = input_channels # if black&white, num_channels=1 || if RGB, num_channels=3
    #     self.output_shape = (input_height - kernel_size + 1, input_width - kernel_size + 1, num_filters)
    #     self.weights_shape = (kernel_size, kernel_size, input_channels, num_filters)
    #     self.weights = np.random.randn(*self.weights_shape)
    #     self.biases = np.random.randn(*self.output_shape)
    #
    # def forward(self, input):
    #     self.input = input
    #     self.output = np.copy(self.biases)
    #     for i in range(self.num_filters):
    #         for j in range(self.num_channels):
    #             # self.output[i] += signal.correlate2d(self.input[j], self.weights[i, j], "valid")
    #             self.output[:, :, i] += signal.correlate2d(self.input[:, :, j], self.weights[:, :, j, i], "valid")
    #     return self.output
    #
    # def backward(self, output_gradient, learning_rate):
    #     weights_gradient = np.zeros(self.weights_shape)
    #     input_gradient = np.zeros(self.input_shape)
    #     for i in range(self.num_filters):
    #         for j in range(self.num_channels):
    #             # weights_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
    #             # input_gradient[j] += signal.convolve2d(output_gradient[i], self.weights[i, j], "full")
    #             weights_gradient[:, :, j, i] = signal.correlate2d(self.input[:, :, j], output_gradient[:, :, i], "valid")
    #             input_gradient[:, :, j] += signal.convolve2d(output_gradient[:, :, i], self.weights[:, :, j, i], "full")
    #
    #     self.weights -= learning_rate * weights_gradient
    #     self.biases -= learning_rate * output_gradient
    #     return input_gradient

    def save_layer(self, csv_writer):
        first_row = [__name__ +'.' + type(self).__name__ +'(' + str(self.num_filters) +', ' + str(self.kernel_size) + ', "' + str(self.initializer) + '")']
        csv_writer.writerow(first_row)

        # Write input channels before anything
        input_channels = self.input.shape[2]
        csv_writer.writerow([input_channels])

        for i in range(input_channels):
            for j in range(self.kernel_size):
                for k in range(self.kernel_size):
                    csv_writer.writerow(self.weights[k, j, i, :])

        for i in range(self.num_filters):
            csv_writer.writerow([self.bias[i]])

    def load_layer(self, csv_reader):
        # Read input channels before anything, then initialize weights
        input_channels = int(next(csv_reader)[0])
        self.weights = np.zeros(shape=(self.kernel_size, self.kernel_size, input_channels, self.num_filters))

        for i in range(input_channels):
            for j in range(self.kernel_size):
                for k in range(self.kernel_size):
                    self.weights[k, j, i, :] = [float(l) for l in next(csv_reader)]

        for i in range(self.num_filters):
            self.bias[i] = float(next(csv_reader)[0])


class Flatten(Layer):
    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        flattened_size = np.prod(self.input_shape)
        return input.reshape((flattened_size, 1))

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)

    def save_layer(self, csv_writer):
        csv_writer.writerow([__name__+'.'+type(self).__name__+'()'])

    def load_layer(self, csv_reader):
        pass


class Softmax(Layer):
    def forward(self, input):
        # Calcular la exponencial de cada elemento y la suma de exponenciales
        exp_input = np.exp(input - np.max(input))  # Evitar problemas de estabilidad numérica
        sum_exp_input = np.sum(exp_input)

        # Calcular las probabilidades de clase
        self.output = exp_input / sum_exp_input
        return self.output

        # tmp = np.exp(input)
        # self.output = tmp / np.sum(tmp)
        # return self.output

    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

    def save_layer(self, csv_writer):
        csv_writer.writerow([__name__+'.'+type(self).__name__+'()'])


class MaxPooling(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input_shape = input.shape
        self.max_locations = []

        input_height, input_width, filters = input.shape

        output_height = 1 + (input_height - self.pool_size) // self.stride
        output_width = 1 + (input_width - self.pool_size) // self.stride

        output = np.zeros(shape=(output_height, output_width, filters))

        for i in range(output_height):
            for j in range(output_width):
                vertical_start = i * self.stride
                vertical_end = vertical_start + self.pool_size
                horizontal_start = j * self.stride
                horizontal_end = horizontal_start + self.pool_size

                pool_region = input[vertical_start:vertical_end, horizontal_start:horizontal_end, :]

                output[i, j, :] = np.amax(pool_region, axis=(0, 1))

                for f in range(filters):
                    max_index = np.unravel_index(np.argmax(pool_region[:, :, f], axis=None), pool_region[:, :, f].shape)
                    self.max_locations.append([vertical_start+max_index[0], horizontal_start+max_index[1], f])

        return output

        # num_filters = input.shape[2] # depth --> number of filters
        # output_height = 1 + (input.shape[0] - self.pool_size) // self.stride
        # output_width = 1 + (input.shape[1] - self.pool_size) // self.stride
        #
        # output = np.zeros(shape=(output_height, output_width, num_filters))
        # self.max_locations = np.zeros_like(input)
        #
        # for i in range(0, output_height):
        #     for j in range(0, output_width):
        #         vertical_start = i * self.stride
        #         vertical_end = vertical_start + self.pool_size
        #         horizontal_start = j * self.stride
        #         horizontal_end = horizontal_start + self.pool_size
        #
        #         pool_region = input[vertical_start:vertical_end, horizontal_start:horizontal_end, :]
        #
        #         max_indices = np.unravel_index(np.argmax(pool_region, axis=None), pool_region.shape)
        #         self.max_locations[vertical_start:vertical_end, horizontal_start:horizontal_end, :] = 0
        #         self.max_locations[vertical_start + max_indices[0], horizontal_start + max_indices[1], :] = 1
        #
        #         output[i, j, :] = np.amax(pool_region, axis=(0, 1))
        #
        # return output

    def backward(self, output_gradient, learning_rate):
        output_height, output_width, filters = output_gradient.shape

        dinput = np.zeros(shape=self.input_shape)

        counter = 0
        for i in range(output_height):
            for j in range(output_width):
                for f in range(filters):
                    i_dinput, j_dinput, f_dinput = self.max_locations[counter]
                    dinput[i_dinput, j_dinput, f_dinput] = output_gradient[i, j, f]
                    counter += 1
        # print(output_gradient[0])
        # print(dinput[0])


        # for i in range(0, len(self.max_locations)):
        #     output_filter_index = self.max_locations[i][0]
        #     output_height_index = 1 + (self.max_locations[i][1] - self.pool_size) // self.stride
        #     output_width_index = 1 + (self.max_locations[i][2] - self.pool_size) // self.stride
        #     dinput[self.max_locations[i]] = output_gradient[output_filter_index, output_height_index, output_width_index]

        return dinput

        # dinput = np.zeros_like(self.max_locations)
        #
        # for c in range(self.max_locations.shape[0]):
        #     for i in range(self.max_locations.shape[1]):
        #         i_doutput = 1 + (i - self.pool_size) // self.stride
        #         for j in range(self.max_locations.shape[2]):
        #             j_doutput = 1 + (j - self.pool_size) // self.stride
        #             if self.max_locations[c, i, j] == 1:
        #                 dinput[c, i, j] = output_gradient[c, i_doutput, j_doutput]
        #
        # return dinput

        # dinput = np.zeros_like(self.max_locations)
        #
        # for i in range(self.max_locations.shape[0]):
        #     i_doutput = 1 + (i - self.pool_size) // self.stride
        #     for j in range(self.max_locations.shape[1]):
        #         j_doutput = 1 + (j - self.pool_size) // self.stride
        #         for c in range(self.max_locations.shape[2]):
        #             if self.max_locations[i, j, c] == 1:
        #                 dinput[i, j, c] = output_gradient[i_doutput, j_doutput, c]
        #
        # return dinput

    def save_layer(self, csv_writer):
        first_row = [__name__+'.'+type(self).__name__+'('+ str(self.pool_size)+', '+ str(self.stride)+')']
        csv_writer.writerow(first_row)

    def load_layer(self, csv_reader):
        pass


class Dropout(Layer):
    def __init__(self, probability, is_training=True):
        self.dropout_probability = probability # probabilidad de que neuronas no mueran
        self.is_training = is_training

    def forward(self, input):
        if self.is_training:
            self.mask = (np.random.rand(*input.shape) < self.dropout_probability) / self.dropout_probability # inverted dropout
            output = input*self.mask
        else:
            output = input

        return output

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient*self.mask
        return input_gradient

    def save_layer(self, csv_writer):
        first_row = [__name__ +'.' + type(self).__name__ + '(' + str(self.dropout_probability) + ', ' + 'False' + ')']
        csv_writer.writerow(first_row)

    def load_layer(self, csv_reader):
        pass
