from network import load_network

# load the already trained network
myNetwork = load_network('C2F_network.csv')

# test
X = [[16.8]]    # z=62.24
z = myNetwork.predict(X)
print(z)
