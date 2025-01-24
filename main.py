import numpy as np
weights = np.around(np.random.uniform(size=6),decimals=2)
'''np.random.uniform() generates random numbers unifromly distributed within the range 0,1 ,size =6 specifies that we want
six random numbers np.around(x,decimal=n) rounds the elements in array x to n decimal places '''
print(weights)
biases = np.around(np.random.uniform(size=3),decimals=2)
'''We have 6 weights and 3 biases, one for each node in the hidden layer as well as for each node in the output layer.'''
print(biases)