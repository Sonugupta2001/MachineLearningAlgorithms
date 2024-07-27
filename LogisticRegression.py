import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression class
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        # Add a column of ones to X for the bias term
        X = np.insert(X, 0, 1, axis=1)
        self.theta = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            predictions = sigmoid(X.dot(self.theta))
            errors = predictions - y
            gradient = X.T.dot(errors) / len(y)
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return sigmoid(X.dot(self.theta)) >= 0.5

# Example usage
X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression(learning_rate=0.01, epochs=1000)
model.fit(X, y)
predictions = model.predict(X)

print("Predictions:", predictions)
