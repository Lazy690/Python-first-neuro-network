import numpy as np

# Input data — 4 examples
inputs = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
])

# Target outputs — what we want the neuron to learn
targets = np.array([0, 1, 1, 0])  # 0 = apple, 1 = banana

# Random weights for 2 inputs
weights = np.random.randn(2)
bias = 0.0

# Learning rate (how much we adjust weights each time)
learning_rate = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

output = sigmoid(np.dot(inputs, weights) + bias)

for epoch in range(1000):  # Train for 1000 loops
    for i in range(len(inputs)):
        # ---- Forward pass ----
        input_i = inputs[i]
        target = targets[i]

        z = np.dot(input_i, weights) + bias   # Linear combo
        prediction = sigmoid(z)               # Activation

        # ---- Calculate error ----
        error = prediction - target

        # ---- Backward pass (gradient descent) ----
        dcost_dpred = error                  # How much error came from the prediction
        dpred_dz = sigmoid_derivative(z)     # How much prediction changes with z
        dz_dw = input_i                      # How z changes with weights

        # Update weights & bias
        weights -= learning_rate * dcost_dpred * dpred_dz * dz_dw
        bias -= learning_rate * dcost_dpred * dpred_dz

# Final weights and bias
print("Trained weights:", weights)
print("Trained bias:", bias)

def predict(x):
    return sigmoid(np.dot(x, weights) + bias)

print("Predict [1, 0]:", predict([1, 0]))
print("Predict [0, 1]:", predict([0, 1]))
print("Predict [1, 1]:", predict([1, 1]))
print("Predict [0, 0]:", predict([0, 0]))
print("Predict [1, 2]:", predict([1, 2]))  # New data!
