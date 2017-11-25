from random import seed

from network import initialize_network, forward_propagate, backward_propagate_error

# test forward propagation
seed(1)
network = initialize_network(2, 1, 2)
output = forward_propagate(network, [1, 0])
print("Layers:")
for layer in network:
    print(layer)
print("Output:\n" + str(output))



# test backward propagation of error
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
        [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)
for layer in network:
    print(layer)