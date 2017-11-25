from random import seed

from network import initialize_network, forward_propagate

seed(1)
network = initialize_network(2, 1, 2)
output = forward_propagate(network, [1, 0])
print("Layers:")
for layer in network:
    print(layer)
print("Output:\n" + str(output))
