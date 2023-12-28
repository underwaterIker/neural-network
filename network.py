import matplotlib.pyplot as plt
import csv
import os
import numpy as np

# Do NOT remove the following imports, they are completely needed!!!
import layers
import activations


def load_network(file_name: str):
    myNetwork = Network()
    folder_name = 'saved_networks'
    file_directory = folder_name + '/' + file_name
    with open(file_directory, 'r') as f:
        csv_reader = csv.reader(f)
        numberOfLayers = int(next(csv_reader)[0])
        for i in range(numberOfLayers):
            # defining the layer
            layer_parameters = next(csv_reader)
            layer = eval(layer_parameters[0])
            layer.load_layer(csv_reader)
            # appending the layer
            myNetwork.layers.append(layer)
    return myNetwork


class Network:
    def __init__(self):
        self.layers = []

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backpropagation(self, average_gradient, learning_rate):
        for layer in reversed(self.layers):
            average_gradient = layer.backward(average_gradient, learning_rate)

    def train(self, loss, loss_prime, x_train, y_train, x_test, y_test, batch_size, epochs=1000, learning_rate=0.01, verbose=True):
        loss_training_plot = []
        accuracy_training_plot = []
        loss_validation_plot = []
        accuracy_validation_plot = []
        epochs_plot = []

        num_samples_training = len(x_train)
        num_samples_validation = len(x_test)
        for epoch in range(epochs):
            # TRAINING PART
            total_loss_training = 0
            total_accuracy_training = 0
            for i in range(0, num_samples_training, batch_size):
                batch_x = x_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                # Inicializar el gradiente promedio para el lote
                average_gradient = 0 # np.zeros_like(batch_y) ???

                for j in range(len(batch_x)):
                    x = batch_x[j]
                    y = batch_y[j]

                    # Forward pass
                    prediction = self.predict(x)

                    if np.argmax(prediction) == np.argmax(y):
                        total_accuracy_training += 1

                    # Calcular la pérdida
                    loss_value = loss(y, prediction)
                    total_loss_training += loss_value

                    # Calcular gradientes
                    gradient = loss_prime(y, prediction)

                    # Acumular gradientes para el lote
                    average_gradient += gradient

                # Promediar el gradiente para el lote
                average_gradient /= len(batch_x)

                # Backward pass al final del lote
                self.backpropagation(average_gradient, learning_rate)

            # Imprimir la pérdida y precisión promedio en cada época
            average_loss_training = total_loss_training / num_samples_training
            average_accuracy_training = (total_accuracy_training / num_samples_training) * 100

            if verbose:
                print(f'Epoch {epoch + 1}/{epochs}, |TRAINING|, Average Loss: {average_loss_training}, Average Accuracy: {average_accuracy_training}%')

            loss_training_plot.append(average_loss_training)
            accuracy_training_plot.append(average_accuracy_training)
            epochs_plot.append(epoch)

            # VALIDATION PART
            total_loss_validation = 0
            total_accuracy_validation = 0
            for i in range(num_samples_validation):
                x = x_test[i]
                y = y_test[i]

                # Forward pass
                prediction = self.predict(x)

                if np.argmax(prediction) == np.argmax(y):
                    total_accuracy_validation += 1

                # Calcular la pérdida
                loss_value = loss(y, prediction)
                total_loss_validation += loss_value

            # Imprimir la pérdida y precisión promedio en cada época
            average_loss_validation = total_loss_validation / num_samples_validation
            average_accuracy_validation = (total_accuracy_validation / num_samples_validation) * 100

            if verbose:
                print(f'Epoch {epoch + 1}/{epochs}, |VALIDATION|, Average Loss: {average_loss_validation}, Average Accuracy: {average_accuracy_validation}%')

            loss_validation_plot.append(average_loss_validation)
            accuracy_validation_plot.append(average_accuracy_validation)

        # loss & accuracy vs epochs plots
        plt.rcParams["figure.figsize"] = (10, 5)

        plt.subplot(121)
        plt.plot(epochs_plot, loss_training_plot, label="Training")
        plt.plot(epochs_plot, loss_validation_plot, label="Validation")
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Average Loss')
        plt.title('Loss')

        plt.subplot(122)
        plt.plot(epochs_plot, accuracy_training_plot, label="Training")
        plt.plot(epochs_plot, accuracy_validation_plot, label="Validation")
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Average Accuracy (%)')
        plt.title('Accuracy')

        plt.suptitle('Training evaluation')

        plt.show()



        # errors_plot = []
        # epochs_plot = []
        # for e in range(epochs):
        #     error = 0
        #     for x, y in zip(x_train, y_train):
        #         # forward
        #         output = self.predict(x)
        #
        #         # error
        #         error += loss(y, output)
        #
        #         # backward
        #         grad = loss_prime(y, output)
        #         for layer in reversed(self.layers):
        #             grad = layer.backward(grad, learning_rate)
        #
        #     error /= len(x_train)
        #     if verbose:
        #         print(f"{e + 1}/{epochs}, error={error}")
        #
        #     # for errors vs epochs plot
        #     errors_plot.append(error)
        #     epochs_plot.append(e)
        #
        # # errors vs epochs plot
        # plt.plot(epochs_plot, errors_plot)
        # plt.xlabel('epochs')
        # plt.ylabel('error')
        # plt.title('error vs epochs')
        # plt.show()

    def save_network(self, file_name: str):
        folder_name = 'saved_networks'
        if os.path.isdir(folder_name) is False:
            os.mkdir(folder_name)

        file_directory = folder_name + '/' + file_name
        with open(file_directory, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([len(self.layers)])
            for layer in self.layers:
                layer.save_layer(csv_writer)
