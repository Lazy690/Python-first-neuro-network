import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(inputs, weights, bias):
    raw = np.dot(inputs, weights) + bias
    probability = sigmoid(raw)
    return probability * 100  # Convert to percentage

# Trained weights and bias
weights = np.array([-0.5, 4.02])
bias = -1.06

# Try some inputs
test_inputs = [
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],
    [1, 2]
]

for input_vector in test_inputs:
    confidence = predict(input_vector, weights, bias)
    print(f"Input {input_vector} → {confidence:.2f}% banana confidence")
