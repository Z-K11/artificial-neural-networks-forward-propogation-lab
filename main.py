import numpy as np
weights = np.around(np.random.uniform(size=6),decimals=2)
'''np.random.uniform() generates random numbers unifromly distributed within the range 0,1 ,size =6 specifies that we want
six random numbers np.around(x,decimal=n) rounds the elements in array x to n decimal places '''
print(weights)
biases = np.around(np.random.uniform(size=3),decimals=2)
'''We have 6 weights and 3 biases, one for each node in the hidden layer as well as for each node in the output layer.'''
print(biases)
'''our neuron has 2 inputs one outputs 6 weights and 3 biases for the hidden layer it two random inputs are '''
x_1 =0.5
x_2 =0.85
print ('x1 is {} and x2 is {}'.format(x_1,x_2))
z_11 = x_1*weights[0]+x_2*weights[1] + biases[0]
z_12 = x_1*weights[2]+x_2*weights[3] + biases[1]
print('the weighted sum of input at the first and second node of the hidden layer is {}  and {}'.format(np.around(
    z_11,decimals=4),np.around(z_12,decimals=4)))
'''assuming a sigmoid function as the activation function let's compute the activation of the first node '''
a_11 = 1.0/(1.0+np.exp(-z_11))
print('activation of the first node in the hidden layer is {}'.format(np.around(a_11,decimals=4)))
a_12 = 1.0/(1.0+np.exp(-z_12))
print('activation of the second node in the hidden layer is {}'.format(np.around(a_12,decimals=4)))
z_2=a_11*weights[4]+a_12*weights[5]+biases[2]
print('the weighted sum at the only node of the output layer is {}'.format(np.around(z_2,decimals=4)))
a_2 = 1.0/(1.0 + np.exp(-z_2))
print('activation of the output layer is {}'.format(np.around(a_2,decimals=4)))
'''although this how a neural network works computing the weighted sum and activations at each node is not efficient
we will automate this process let's first define the structure of our neural network'''
n=2 # number of input
num_of_hidden_layers =2
m=[2,2] # number of nodes in the hidden layer
'''m is a list so that each index has the number of nodes for each layer we have two layers so we have two elements in the 
list '''
num_of_nodes_output=1
num_of_previous_nodes = n 
network = {}
for layer in range(num_of_hidden_layers+1):
    if layer == num_of_hidden_layers:
        layer_name ='output'
        num_nodes = num_of_nodes_output
    else:
        layer_name='layers_{}'.format(layer+1)
        num_nodes=m[layer]
    network[layer_name]={}
    for node in range(num_nodes):
        node_name ='node_{}'.format(layer+1)
        network[layer_name][node_name]={'weights':np.around(np.random.uniform(size=num_of_previous_nodes),decimals=2),
                                        'bias':np.around(np.random.uniform(size=1),decimals=2)}
    num_of_previous_nodes=num_nodes
print(network)
def initialize_network(num_inputs,num_hidden_layers,num_nodes_hidden,num_output_nodes):
    num_nodes_previous = num_inputs
    network={}
    for layer in range(num_hidden_layers+1):
        if layer ==num_hidden_layers:
            layer_name='output_layer'
            num_nodes=num_output_nodes
        else:
            layer_name='layer_{}'.format(layer+1)
            num_nodes=num_nodes_hidden[layer]
        network[layer_name]={}
        for node in range(num_nodes):
            node_name='node_{}'.format(node +1)
            network[layer_name][node_name]={'weights':np.around(np.random.uniform(size=num_nodes_previous),decimals=4),
                                        'bias':np.around(np.random.uniform(size=1),decimals=4)}
        num_nodes_previous=num_nodes
    return network
def compute_weighted_sum(input,weights,bias):
    return np.sum(input*weights)+bias
def node_activation(weighted_sum):
    return 1.0/(1.0 + np.exp(-1*weighted_sum))
from random import seed
np.random.seed(12)
inputs= np.around(np.random.uniform(size=5),decimals=4)
print('the input to the networks are {}'.format(inputs))
network = initialize_network(5,3,[3,2,3],1)
node_weights = network['layer_1']['node_1']['weights']
node_bias = network['layer_1']['node_1']['bias']
weighted_sum = compute_weighted_sum(inputs,node_weights,node_bias)
print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0], decimals=4)))
activation = node_activation(weighted_sum)
print('The output of the first node in the hidden layer is {}'.format(np.around(activation[0], decimals=4)))
