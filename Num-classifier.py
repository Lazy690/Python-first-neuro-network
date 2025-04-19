import numpy as np

# Data
three_a = np.array([[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 67,255,255,255,255,255,255,255,
 255,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  67,255,255,255,255,255,255,255,255,255,255, 67,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0, 67,255,255,255,255,255,255,255,255,255,
 255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,
 255,255,255,255,255,105, 23,  0,255,255,255,255,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0, 52,255,255,255, 67,  0,  0,  0,  0,  0,255,
 255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,167,255,
 255,255,  0,  0,  0,  0,  0,  0,119,255,255,255,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0, 85,255,255, 85,  0,  0,  0,  0,  0,  0,157,
 255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 11, 36,
  36, 11,  0,  0,  0,  0,  0,  0,255,255,255,255,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 23,105,255,255,255,
 255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,255,255,255,255,255,255,255, 67,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,255,
 255,255,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,133,255,255,255,255,255,255, 85,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 23,105,255,255,
 255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0, 23,255,255,255,255, 23,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 67,
 255,255,255,105,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,145,255,
 255,133,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,255,255,255,255, 85,  0,  0,  0,  0,  0,  0,
 255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,
 255,255,255,157, 86, 86,157,255,255,255,255,255,255,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0, 23,255,255,255,255,255,255,255,255,255,255,
 255,255,255, 67,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 23,
 186,255,255,255,255,255,255,255,255,255,133,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 67,255,255,255,255,255,255,118,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0]])  # Class 0, 3s

seven_a = np.array([[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 20,213,255,
 255,255,255,255,255,255,255,255,255,255,255,255, 67,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0, 47,255,255,255,255,255,255,255,255,255,255,255,
 255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 26,255,255,
 255,255,255,255,255,255,255,255,255,255,255,255,255,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  3, 26, 66, 84, 86, 86, 86, 86, 86, 86, 86,102,
 255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,133,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 67,255,
 255,255,133,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0, 23,255,255,255,255,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,158,255,255,
 255, 67,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0, 67,255,255,255,158,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,
  23,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,255,255,255,255, 67,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,133,255,255,255,255,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0, 85,255,255,255,255,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,167,255,255,255,255,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  23,255,255,255,255,145,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,255,255,146,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,167,
 255,255,255,255, 23,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,167,255,255,255,255, 67,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,255,255,
 255,255, 67,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,255,255,255,119,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,145,255,
 255, 23,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0]])  # Class 1, 7s

three_a = three_a / 255.0
seven_a = seven_a / 255.0

inputs = np.array([three_a.flatten(), seven_a.flatten()])
labels = np.array([[1, 0],  # 3
                   [0, 1]]) # 7

# Weights
hidden_weights = np.random.randn(20, 729)
hidden_biases = np.random.randn(20)
output_weights = np.random.randn(2, 20)
output_biases = np.random.randn(2)

# Hyperparameters
learning_rate = 0.1
epochs = 5000

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

def print_ready_to_use_weights():
    print("hidden_weights = np.array(" + np.array2string(hidden_weights, separator=",", threshold=np.inf) + ")")
    print("hidden_biases = np.array(" + np.array2string(hidden_biases, separator=",", threshold=np.inf) + ")")
    print("output_weights = np.array(" + np.array2string(output_weights, separator=",", threshold=np.inf) + ")")
    print("output_biases = np.array(" + np.array2string(output_biases, separator=",", threshold=np.inf) + ")")

print_ready_to_use_weights()

def print_prediction(pattern_name, pattern, hidden_weights, hidden_biases, output_weights, output_biases):
    test = pattern.flatten()
    hidden = relu(np.dot(hidden_weights, test) + hidden_biases)
    output = softmax(np.dot(output_weights, hidden) + output_biases)
    print(f"{pattern_name} prediction: {output[0]*100:.2f}% 3, {output[1]*100:.2f}% 7")

print_prediction("number 3", three_a, hidden_weights, hidden_biases, output_weights, output_biases)
print_prediction("number 7", seven_a, hidden_weights, hidden_biases, output_weights, output_biases)

