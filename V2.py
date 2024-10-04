# Biblioteke
from DZ2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # ili standradizacija zbog sigmoid

# Ovde jedino sto je ispravljeno je da je grafik pravilno okrenut
# Program puca kod exp(-z)
# Tacnost je jako losa

class LogistickaRegresija():
    def __init__(self, lr, n_iters):
        self.lr = lr
        self.n_iters = n_iters
        self.thetas = None  # Inicijalno theta parametri nisu postavljeni

    # Sigmoidna funkcija
    def sigmoid(self, X, theta):
        z = X @ theta
        s = 1 / (1 + np.exp(-z))
        return s

    # Kriterijumska funkcija
    def cost(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(X, theta)
        epsilon = 1e-10  # Mali epsilon da bi se izbeglo deljenje sa nulom
        cost = (-1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost

    # Gradijentni spust
    def gradient_descent(self, X, y):
        m, n = X.shape
        theta = np.zeros(n)  # Inicijalizujemo theta parametre na nule
        cost_all = np.empty(self.n_iters)

        for i in range(self.n_iters):
            h = self.sigmoid(X, theta)
            dJ = (1 / m) * X.T @ (h - y)  # Racunanje gradijenta kriterijumske funkcije
            theta -= self.lr * dJ  # Azuriranje parametara theta
            cost = self.cost(X, y, theta)  # Kriterijumska funkcija u svakoj iteraciji
            if i % 100 == 0:  # Printanje cost-a na svakih 100 iteracija
                print(f"Iteracija {i}: Cost = {cost}")
            cost_all[i] = cost

        return theta, cost_all

    # Treniranje modela za sve klase
    def fit(self, X, y):
        unique_classes = np.unique(y)
        self.thetas = []

        for cls in unique_classes:
            y_binary = np.where(y == cls, 1, 0)  # "One-vs-all" oznake
            theta, cost_all = self.gradient_descent(X, y_binary)
            self.thetas.append(theta)
            self.plot_cost(cost_all, cls)

    # Predikcija za sve klase
    def predict(self, X):
        m = X.shape[0]
        all_probs = np.zeros((m, len(self.thetas)))

        for i, theta in enumerate(self.thetas):
            all_probs[:, i] = self.sigmoid(X, theta)

        y_pred = np.argmax(all_probs, axis=1)
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

# Pretpostavimo da su X_train i y_train već definisani
# Treniranje modela
lr.fit(X_train.to_numpy(), y_train.to_numpy())

# Plotovanje vrednosti kriterijumske funkcije
# (Plot će se pojaviti za svaku klasu unutar fit funkcije)

# Predikcije na trening skupu
y_train_pred = lr.predict(X_train.to_numpy())
print("Predikcije na trening skupu:", y_train_pred)

# Predikcije na test skupu
y_test_pred = lr.predict(X_test.to_numpy())
print("Predikcije na test skupu:", y_test_pred)

# Tacnost na trening skupu
train_accuracy = lr.accuracy(y_train.to_numpy(), y_train_pred)
print("Tacnost na trening skupu:", train_accuracy)

# Tacnost na test skupu
test_accuracy = lr.accuracy(y_test.to_numpy(), y_test_pred)
print("Tacnost na test skupu:", test_accuracy)
