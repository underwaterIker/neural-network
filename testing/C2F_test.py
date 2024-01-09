from network import load_network

# load the already trained network
myNetwork = load_network('C2F_network.csv')

# test
X = [[16.8], [4.3], [-8.6], [-14], [27]]
for i in X:
    pred = myNetwork.predict(i)
    print(pred)
