# Biblioteke
from DZ2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit # ili standradizacija zbog sigmoid
# nije dobro

class LogistickaRegresija():
    def __init__(self, lr, n_iters):
        self.lr = lr
        self.n_iters = n_iters
        self.theta = None  # Inicijalno theta parametri nisu postavljeni

    # Sigmoidna funkcija
    def sigmoid(self, X):
        z = X @ self.theta
        s = 1 / (1 + np.exp(-z))
        return s

    # Kriterijumska funkcija
    def cost(self, X, y):
        m = len(y)
        h = self.sigmoid(X)
        epsilon = 1e-10  # Mali epsilon da bi se izbeglo deljenje sa nulom
        cost = (-1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost

    # Gradijentni spust
    def gradient_descent(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)  # Inicijalizujemo theta parametre na nule
        cost_all = np.empty((1, self.n_iters))

        for i in range(0, self.n_iters):
            h = self.sigmoid(X)
            dJ = (1 / m) * X.T @ (h - y)  # Racunanje gradijenta kriterijumske funkcije
            self.theta = self.theta - self.lr * dJ  # Azuriranje parametara theta
            cost = self.cost(X, y)  # Kriterijumska funkcija u svakoj iteraciji
            print(f"Iteracija {i}: Cost = {cost}")
            cost_all[0, i] = cost

        return self.theta, cost_all

    def predict(self, X, theta):
        m = X.shape[0]
        y_pred = np.empty(m, dtype=int)  # Inicijalizacija niza za cuvanje predvidjenih oznaka

        for i in range(m):
            res = self.sigmoid(X[i, :])  # Racunanje verovatnoce za svaku klasu
            y_pred[i] = 1 if res >= 0.5 else 0  # Ako je verovatnoca veca od 0.5, predvidjena klasa je 1, inace 0

        return y_pred

    def accuracy(self, y_true, y_pred):
        correct_predictions = np.sum(y_pred == y_true)  # Broj tacnih predikcija
        total_examples = len(y_true)  # Broj primera
        accuracy = correct_predictions / total_examples
        return accuracy

    def plot_cost(self, cost_all, cls):
        iterations = len(cost_all)
        plt.plot(np.arange(0, iterations), cost_all)
        plt.xlabel('Iteracije')
        plt.ylabel('Cost')
        plt.title(f'Cost vs. Iteracije za klasu {cls}')
        plt.gca().invert_yaxis()  # Invertovanje y ose za pravilno prikazivanje opadanja cost funkcije
        plt.show()


# Instanciranje klase LogistickaRegresija
lr = LogistickaRegresija(lr=0.01, n_iters=1000)

# Treniranje modela
theta, costs = lr.gradient_descent(X_train.to_numpy(), y_train.to_numpy())

# Plotovanje vrednosti kriterijumske funkcije
lr.plot_cost(costs)

# Predikcije na trening skupu
y_train_pred = lr.predict(X_train.to_numpy(), theta)
print("Predikcije na trening skupu:", y_train_pred)

# Predikcije na test skupu
y_test_pred = lr.predict(X_test.to_numpy(), theta)
print("Predikcije na test skupu:", y_test_pred)

# Tacnost na trening skupu
train_accuracy = lr.accuracy(y_train.to_numpy(), y_train_pred)
print("Tacnost na trening skupu:", train_accuracy)

# Tacnost na test skupu
test_accuracy = lr.accuracy(y_test.to_numpy(), y_test_pred)
print("Tacnost na test skupu:", test_accuracy)
