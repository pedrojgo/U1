import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lee el conjunto de datos desde un archivo CSV
df = pd.read_csv("mi_archivo.csv")

# Codifica las etiquetas de clase 'Manzana', 'Pera' y 'Banana' a valores numéricos (por ejemplo, 0, 1 y 2)
df['Tipo_de_Fruta'] = df['Tipo_de_Fruta'].map({'Manzana': 0, 'Pera': 1, 'Banana': 2})

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = df[['Precio', 'Peso']].values  # Usamos Precio y Peso como características
y = df['Tipo_de_Fruta'].values     # La etiqueta de clase es 'Tipo_de_Fruta'

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 0, 2)  # Usamos 0 para Manzana y 2 para Pera

# Crear una instancia del perceptrón y entrenarlo
perceptron = Perceptron(eta=0.1, n_iter=100)
perceptron.fit(X, y)

# Crear un gráfico de dispersión para mostrar la clasificación
colors = ['red', 'green', 'blue']
markers = ['o', 's', 'x']
for tipo in np.unique(y):
    plt.scatter(X[y == tipo][:, 0], X[y == tipo][:, 1], color=colors[tipo], marker=markers[tipo], label=tipo)

# Realizar predicciones
nuevo_dato = np.array([2.3, 155])  # Ejemplo de un nuevo dato de Precio y Peso de una fruta
prediccion = perceptron.predict(nuevo_dato)

# Convertir la predicción a tipo de fruta
tipos_de_fruta = ['Manzana', 'Pera', 'Banana']
tipo_predicho = tipos_de_fruta[prediccion]

plt.scatter(nuevo_dato[0], nuevo_dato[1], color='purple', marker='*', s=200, label=f'Predicción: {tipo_predicho}')
plt.xlabel('Precio')
plt.ylabel('Peso')
plt.legend()
plt.show()
