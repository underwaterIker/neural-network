import numpy as np
import cv2
import zipfile
import os
import matplotlib.pyplot as plt
from PIL import Image

from layers import Dense, Convolutional, Flatten, MaxPooling, Dropout, Softmax
from activations import Sigmoid, ReLU
from losses import binary_cross_entropy, binary_cross_entropy_prime, mse, mse_prime
from network import Network


def preprocess_data(datos, image_size):
    datos_entrenamiento = []
    x = []
    y = []
    for i, (imagen, etiqueta) in enumerate(datos['train']):
        imagen = cv2.resize(imagen.numpy(), (image_size, image_size))
        imagen = imagen.reshape((image_size, image_size, 3))
        datos_entrenamiento.append([imagen, etiqueta])

        x.append(imagen)
        y.append(etiqueta)
    x = np.array(x).astype(float)/255
    y = np.array(y)

    return x, y


# load Dataset
if os.path.exists('../datasets/cats_vs_dogs_Dataset') is False:
    with zipfile.ZipFile('../datasets/cats_vs_dogs_Dataset.zip', 'r') as f:
        f.extractall(path='../datasets/')

# img = Image.open('../datasets/cats_vs_dogs_Dataset/Cat/0.jpg')
# # img.load()
# data = np.asarray(img, dtype='int32')
# print(data.shape)
# plt.figure()
# plt.imshow(img)
# plt.show()
# cat_img = cv2.imread('../datasets/cats_vs_dogs_Dataset/Cat/0.jpg')
# print('jjjjjjjjjjjjjjjjjjjjjjjjjjjj')
# print(cat_img.shape)

x = []
y = []
for i, (image) in enumerate(os.listdir('../datasets/cats_vs_dogs_Dataset/Cat')):
    if i == 100:
        break
    try:
        cat_img = cv2.imread('../datasets/cats_vs_dogs_Dataset/Cat/'+image, cv2.IMREAD_GRAYSCALE)
        cat_img = cv2.resize(cat_img, (128, 128))
        cat_img = cat_img.reshape((128, 128, 1))

        # plt.figure()
        # plt.imshow(cat_img, cmap='gray')
        # plt.show()

        x.append(cat_img)
        y.append([[0], [1]])
    except:
        # print('cat image error')
        pass

    try:
        dog_img = cv2.imread('../datasets/cats_vs_dogs_Dataset/Dog/' + image, cv2.IMREAD_GRAYSCALE)
        dog_img = cv2.resize(dog_img, (128, 128))
        dog_img = dog_img.reshape((128, 128, 1))

        # plt.figure()
        # plt.imshow(dog_img, cmap='gray')
        # plt.show()

        x.append(dog_img)
        y.append([[1], [0]])
    except:
        # print('dog image error')
        pass
x_train = np.array(x).astype(float)/255
y_train = np.array(y)
# shuffle
p = np.random.permutation(len(x_train))
x_train, y_train = x_train[p], y_train[p]


x = []
y = []
for i, (image) in enumerate(reversed(os.listdir('../datasets/cats_vs_dogs_Dataset/Cat'))):
    if i == 25:
        break
    try:
        cat_img = cv2.imread('../datasets/cats_vs_dogs_Dataset/Cat/'+image, cv2.IMREAD_GRAYSCALE)
        cat_img = cv2.resize(cat_img, (128, 128))
        cat_img = cat_img.reshape((128, 128, 1))

        # plt.figure()
        # plt.imshow(cat_img, cmap='gray')
        # plt.show()

        x.append(cat_img)
        y.append([[0], [1]])
    except:
        # print('cat image error')
        pass

    try:
        dog_img = cv2.imread('../datasets/cats_vs_dogs_Dataset/Dog/' + image, cv2.IMREAD_GRAYSCALE)
        dog_img = cv2.resize(dog_img, (128, 128))
        dog_img = dog_img.reshape((128, 128, 1))

        # plt.figure()
        # plt.imshow(dog_img, cmap='gray')
        # plt.show()

        x.append(dog_img)
        y.append([[1], [0]])
    except:
        # print('dog image error')
        pass
x_validation = np.array(x).astype(float)/255
y_validation = np.array(y)
# shuffle
p = np.random.permutation(len(x_validation))
x_validation, y_validation = x_validation[p], y_validation[p]


print('Dataset loaded!')



# neural network
network = Network()
network.layers = [
    Convolutional(num_filters=16, kernel_size=3, initializer='he'),
    ReLU(),
    MaxPooling(2, 2),
    # Convolutional(num_filters=32, kernel_size=3, initializer='he'),
    # ReLU(),
    # MaxPooling(2, 2),
    # Convolutional(num_filters=64, kernel_size=3, initializer='he'),
    # ReLU(),
    # MaxPooling(2, 2),
    # Dropout(0.8),
    Flatten(),
    Dense(150, initializer='he'),
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
    x_validation,
    y_validation,
    batch_size=1,
    epochs=20,
    learning_rate=0.0001
)
network.save_network('CatsVsDogs_network.csv')
