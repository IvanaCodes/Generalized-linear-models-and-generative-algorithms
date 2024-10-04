# Importovanje biblioteka
import pandas as pd
import numpy as np

# Ucitavanje podataka
data = pd.read_csv("multiclass_data.csv", header=None)
print(data)
print("Broj vrsta i kolona:", data.shape)

# Broj primera: m = 178, broj prediktora: 5, poslednja kolona je labela

# Matrica odlika (prediktora)
X = data.iloc[:, 0:5]
X.insert(0, "Bias", 1)  # Dodavanje kolone za bias matrici odlika
print ("Prikaz prediktora\n", X)
print("Dimenzije matrice prediktora:", X.shape)

# Ciljna promenljiva
y = data.iloc[:, -1]
print ("Ciljna promenljiva\n", y)
print("Dimenzije vektora y:", y.shape)

# Podela obucavajuceg skupa na test i training skupove
split_index = int(0.7 * len(data))
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# Ispis train i test promenljivih
print("X_train", X_train)
print("X_test", X_test)
print("y_train", y_train)
print("y_test", y_test)



