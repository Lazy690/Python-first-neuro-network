import numpy as np

input = np.array([3, 0])  # size = 5, color = red

weights = np.array([
    [0.7, 1.2],   # Apple detector
    [0.4, 0.1],   # Banana detector
    [0.2, 0.9]    # Grape detector
])

biases = np.array([-4, -1, -2])  # Each neuron gets its own bias

output = np.dot(weights, input) + biases

def ReLU(x):  
    return np.maximum(0, x)

ReluResult = ReLU(output)


print("Raw output:", output)
print("ReLU output:", ReluResult)
