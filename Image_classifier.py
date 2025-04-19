import numpy as np

# Data
x_pattern = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]])  # Class 0

i_pattern = np.array([[0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0]])  # Class 1

inputs = np.array([x_pattern.flatten(), i_pattern.flatten()])
labels = np.array([[1, 0],  # X
                   [0, 1]]) # I

# Weights
hidden_weights = np.random.randn(4, 9)
hidden_biases = np.random.randn(4)
output_weights = np.random.randn(2, 4)
output_biases = np.random.randn(2)

# Hyperparameters
learning_rate = 0.1
epochs = 1000

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Stability trick
    return e_x / e_x.sum(axis=0)


# Training
for epoch in range(epochs):
    total_loss = 0

    for x, y_true in zip(inputs, labels):
        ### ---- FORWARD PASS ---- ###
        x = x.reshape(-1)  # Flatten

        hidden_input = np.dot(hidden_weights, x) + hidden_biases
        hidden_output = relu(hidden_input)

        final_input = np.dot(output_weights, hidden_output) + output_biases
        y_pred = softmax(final_input)

        ### ---- LOSS (Cross-Entropy) ---- ###
        loss = -np.sum(y_true * np.log(y_pred + 1e-7))  # Avoid log(0)
        total_loss += loss

        ### ---- BACKPROPAGATION ---- ###
        # Output layer error
        d_output = y_pred - y_true  # shape (2,)

        # Gradient for output weights and biases
        d_output_weights = np.outer(d_output, hidden_output)  # (2,4)
        d_output_biases = d_output

        # Hidden layer error
        d_hidden = np.dot(output_weights.T, d_output) * relu_derivative(hidden_input)

        d_hidden_weights = np.outer(d_hidden, x)
        d_hidden_biases = d_hidden

        ### ---- UPDATE ---- ###
        output_weights -= learning_rate * d_output_weights
        output_biases  -= learning_rate * d_output_biases
        hidden_weights -= learning_rate * d_hidden_weights
        hidden_biases  -= learning_rate * d_hidden_biases

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

def print_prediction(pattern_name, pattern, hidden_weights, hidden_biases, output_weights, output_biases):
    test = pattern.flatten()
    hidden = relu(np.dot(hidden_weights, test) + hidden_biases)
    output = softmax(np.dot(output_weights, hidden) + output_biases)
    print(f"{pattern_name} prediction: {output[0]*100:.2f}% X, {output[1]*100:.2f}% I")

print_prediction("X pattern", x_pattern, hidden_weights, hidden_biases, output_weights, output_biases)
print_prediction("I pattern", i_pattern, hidden_weights, hidden_biases, output_weights, output_biases)

