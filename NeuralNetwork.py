import numpy as np

# Sigmoid and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Weights initialization
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def fit(self, X, y):
        for _ in range(self.epochs):
            # Forward propagation
            hidden_input = np.dot(X, self.weights_input_hidden)
            hidden_output = sigmoid(hidden_input)

            final_input = np.dot(hidden_output, self.weights_hidden_output)
            final_output = sigmoid(final_input)

            # Backward propagation
            output_error = y - final_output
            output_delta = output_error * sigmoid_derivative(final_output)

            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

            # Update weights
            self.weights_hidden_output += hidden_output.T.dot(output_delta) * self.learning_rate
            self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate

    def predict(self, X):
        hidden_output = sigmoid(np.dot(X, self.weights_input_hidden))
        final_output = sigmoid(np.dot(hidden_output, self.weights_hidden_output))
        return final_output

    def accuracy(self, X, y):
        predictions = self.predict(X) >= 0.5
        return np.mean(predictions == y)

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1, epochs=10000)
nn.fit(X, y)

print("Predictions:", nn.predict(X))
print("Accuracy:", nn.accuracy(X, y))
