import numpy as np

input = np.array([1.5, -2.0, 3.0])

weights = np.array([
    [0.2, 0.9, 0.3],
    [0.6, 1.0, -0.2]
])

biases = np.array([1, 2])

outoutput = np.dot(weights, input)
print(outoutput)
output = np.dot(weights, input) + biases

def ReLU(x):  
    return np.maximum(0, x)

ReluResult = ReLU(output)

print("Raw output:", output)
print("ReLU output:", ReluResult)