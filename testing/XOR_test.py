import matplotlib.pyplot as plt
import numpy as np

from network import load_network

# load the already trained network
myNetwork = load_network('XOR_network.csv')

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = myNetwork.predict([[x], [y]])
        points.append([x, y, z[0, 0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()

# test
X = [[0], [0]]
z = myNetwork.predict(X)
print(z)
