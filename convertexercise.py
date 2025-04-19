import numpy as np

inputs = np.array([0., 1., 0., 1., 0., 1., 0., 1., 0.])

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


#[1. 0. 1. 0. 1. 0. 1. 0. 1.]
targets = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])  # 0 = ex, 1 = cross


hidden_weights = np.random.randn(4, 9)
hidden_biases = np.random.randn(4)



hidden_output = relu(np.dot(hidden_weights, inputs) + hidden_biases)

output_weights = np.random.randn(4, 9)
output_biases = np.random.randn(4)

final_output = sigmoid(np.dot(inputs, output_weights) + output_biases)

learning_rate = 0.1

print(hidden_output)
print(final_output)

