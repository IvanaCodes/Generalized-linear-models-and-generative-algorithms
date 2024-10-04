import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import numpy as np
import pandas as pd
from DZ2 import X_train, y_train, X_test, y_test


# Ovaj kod daje najbolje rezultate

class LogistickaRegresija:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.thetas = []  # Will store theta parameters for each class
        self.scaler = StandardScaler()

# Sigmoid funkcija
    def sigmoid(self, X, theta):
        z = X @ theta
        return expit(z)  # Using expit function from scipy.special

# logistički gubitak na osnovu predikcija modela i stvarnih vrednosti
    def cost(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(X, theta)
        epsilon = 1e-10  # Small epsilon to avoid division by zero
        cost = (-1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost

# Gradijentni spust za optimizaciju parametara theta. Iterativno ažurira parametre theta s ciljem minimizacije kriterijumske funkcije
    def gradient_descent(self, X, y):
        m, n = X.shape
        theta = np.zeros(n)  # Initialize theta parameters to zeros
        cost_all = np.empty(self.n_iters)

        for i in range(self.n_iters):
            h = self.sigmoid(X, theta)
            dJ = (1 / m) * X.T @ (h - y)  # Compute gradient of cost function
            theta -= self.lr * dJ  # Update theta parameters
            cost = self.cost(X, y, theta)  # Compute cost function in each iteration
            cost_all[i] = cost

        return theta, cost_all

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)  # Scale the input features
        unique_classes = np.unique(y)

        for cls in unique_classes:
            y_binary = np.where(y == cls, 1, 0)  # Convert to binary class labels for this class
            # Ova linija koristi np.where funkciju da pretvori vektor stvarnih klasa y u binarni vektor y_binary za trenutnu klasu cls
            # Elementi u y_binary će biti 1 gde god je klasa jednaka cls, i 0 svuda drugde.
            theta, costs = self.gradient_descent(X_scaled, y_binary)
            self.thetas.append(theta)
            self.plot_cost(costs, cls)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)  # Scale the input features
        m = X_scaled.shape[0]
        num_classes = len(self.thetas)
        all_probs = np.zeros((m, num_classes))

        for i, theta in enumerate(self.thetas):
            all_probs[:, i] = self.sigmoid(X_scaled, theta)

        y_pred = np.argmax(all_probs, axis=1)
        return y_pred

    def accuracy(self, y_true, y_pred):
        correct_predictions = np.sum(y_pred == y_true)  # Number of correct predictions
        total_examples = len(y_true)  # Number of examples
        accuracy = correct_predictions / total_examples
        return accuracy

# Grafikoni funkcije cost tokom iteracija gradijentnog spusta za svaku klasu
    def plot_cost(self, cost_all, cls):
        iterations = len(cost_all)
        plt.plot(np.arange(0, iterations), cost_all)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title(f'Cost vs. Iterations for class {cls}')
        plt.gca().invert_yaxis()  # Invert y-axis for proper cost function visualization
        plt.show()

# Example usage:
lr = LogistickaRegresija(lr=0.01, n_iters=1000)

# Train the model
lr.fit(X_train.to_numpy(), y_train.to_numpy())

# Predictions on the training set
y_train_pred = lr.predict(X_train.to_numpy())
print("Predictions on the training set:", y_train_pred)

# Predictions on the test set
y_test_pred = lr.predict(X_test.to_numpy())
print("Predictions on the test set:", y_test_pred)

# Accuracy on the training set
train_accuracy = lr.accuracy(y_train.to_numpy(), y_train_pred)
print("Accuracy on the training set:", train_accuracy)

# Accuracy on the test set
test_accuracy = lr.accuracy(y_test.to_numpy(), y_test_pred)
print("Accuracy on the test set:", test_accuracy)
