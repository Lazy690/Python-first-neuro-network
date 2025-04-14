import numpy as np

input = np.array([-5,6,7])

weights = np.array([[0.2,0.9,0.3],
                    [0.6,1,-0.2],
                    [0.7,-0.9,0.5]])

biases = np.array([1,2,3])

output = np.dot(weights, input) + biases

def ReLU(x):  
    return np.maximum(0, x)

ReluResult = ReLU(output)

print("Raw output", output)
print("ReLU output", ReluResult)


