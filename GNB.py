from DZ2 import *
# Ovaj model na najverovatnije otisao u overfitting jer je velika tacnost na train skupu a mala na test

class GaussianNaiveBayes:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.priors = {}
        self.parameters = {}

        # Racunanje a priori verovatnoce za svaku klasu
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        for c, count in enumerate(class_counts):
            self.priors[c] = count / total_samples

    # Računanje srednje vrednosti i varijanse za svaku karakteristiku u svakoj klasi
        for c in self.classes:
            X_c = X_train[y_train == c]
            self.parameters[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0)
            }

    # Gustina verovatnoce
    def _calculate_likelihood(self, mean, var, x):
        epsilon = 1e-9  # Small epsilon value to avoid division by zero
        var += epsilon
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var))

    # Predikcije
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            posteriors = []
            for c in self.classes:
                prior = self.priors[c]
                likelihood = np.sum(np.log(self._calculate_likelihood(self.parameters[c]['mean'], self.parameters[c]['var'], x)))
                posterior = np.log(prior) + likelihood
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

# Instantiranje Gaussian Naive Bayes klasifikatora
gnb = GaussianNaiveBayes()

# Treniranje klasifikatora na training skupu
gnb.fit(X_train.to_numpy(), y_train.to_numpy())

# Predikcije i tacnost na training skupu
train_predictions = gnb.predict(X_train.to_numpy())
train_accuracy = np.mean(train_predictions == y_train)
print("Train Accuracy:", train_accuracy)

# Predikcije i tacnost na test skupu
test_predictions = gnb.predict(X_test.to_numpy())
test_accuracy = np.mean(test_predictions == y_test)
print("Test Accuracy:", test_accuracy)
