import numpy as np

#the 5 means size, and the last number is either 1 or 0, 1 being red and 0 being orage. the color of the fruit I mean.
input = np.array([5, 1])

weights = np.array([0.7, 1.2])

biases = np.array([-4])




output = np.dot(weights, input) + biases

def ReLU(x):  
    return np.maximum(0, x)

ReluResult = ReLU(output)

print("Raw output:", output)
print("ReLU output:", ReluResult)

if ReluResult > 0:
    print("It is an apple")
else:
    print("It is not an apple")