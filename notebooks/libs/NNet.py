#
#   Neural Nets for Teaching (versión PyTorch)
#
#   Cayetano Guerra Artal (c) 2026
#
#   Uso:
#       from NNet import NNet
#       red = NNet([4, 5, 3])      # 4 entradas, 5 neuronas ocultas, 3 salidas
#       red.epocas = 300           # cuántas veces recorremos los datos
#       red.entrenar(X, y)         # X: entradas, y: clases (0, 1, 2...)
#       red.predecir(X)            # devuelve la clase predicha
#


import numpy as np
import torch
from torch import nn


# Funciones de activación disponibles, por su nombre.
activaciones = {
    "sigmoide": nn.Sigmoid,
    "relu": nn.ReLU,
    "tanh": nn.Tanh
}


class NNet:
    """ Red neuronal sencilla totalmente conectada.

        capas: lista con el número de neuronas de cada capa, empezando por
               el número de entradas. Ej.: [4, 5, 3]
    """

    def __init__(self, capas, activacion="sigmoide"):
        self.capas = capas
        self.activacion = activacion

        # Parámetros de entrenamiento: se pueden cambiar antes de entrenar.
        self.epocas = 100
        self.tasa_aprendizaje = 0.3
        self.tam_lote = 15

        # Aquí guardaremos el error de cada época.
        self.historial = []

        self.modelo = self.construir_modelo()

    def construir_modelo(self):
        # Cada par de capas consecutivas se une con una capa Linear
        # (pesos + sesgos) seguida de su función de activación.
        piezas = []
        for entradas, neuronas in zip(self.capas[:-1], self.capas[1:]):
            piezas.append(nn.Linear(entradas, neuronas))
            piezas.append(activaciones[self.activacion]())
        return nn.Sequential(*piezas)

    @staticmethod
    def a_tensor(x):
        # Acepta listas, arrays de NumPy o tensores.
        return torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def a_one_hot(self, y):
        # Si y trae las clases como números enteros (0, 1, 2...), las pasamos
        # a one-hot. Si ya viene en one-hot, la dejamos como está.
        y = torch.as_tensor(np.asarray(y))
        if y.dim() == 1:
            y = nn.functional.one_hot(y.long(), num_classes=self.capas[-1])
        return y.float()

    @staticmethod
    def calcular_error(salidas, objetivos):
        # Suma de errores al cuadrado de cada muestra, promediada en el lote.
        return ((salidas - objetivos) ** 2).sum(dim=1).mean()

    def entrenar(self, x, y, mostrar_cada=None):
        x = self.a_tensor(x)
        y = self.a_one_hot(y)
        if mostrar_cada is None:
            mostrar_cada = max(1, self.epocas // 10)

        optimizador = torch.optim.SGD(self.modelo.parameters(), lr=self.tasa_aprendizaje)
        n = len(x)

        for epoca in range(1, self.epocas + 1):
            # Barajamos los datos al comienzo de cada época.
            orden = torch.randperm(n)

            # Recorremos los datos en mini-lotes.
            for inicio in range(0, n, self.tam_lote):
                indices = orden[inicio:inicio + self.tam_lote]
                entradas, objetivos = x[indices], y[indices]

                optimizador.zero_grad()                         # 1. Borrar gradientes anteriores
                salidas = self.modelo(entradas)                 # 2. Calcular la salida de la red
                error = self.calcular_error(salidas, objetivos) # 3. Medir el error
                error.backward()                                # 4. Calcular gradientes
                optimizador.step()                              # 5. Ajustar pesos y sesgos

            # Al final de la época medimos el error con todos los datos.
            with torch.no_grad():
                error_epoca = self.calcular_error(self.modelo(x), y).item()
            self.historial.append(error_epoca)

            if epoca == 1 or epoca % mostrar_cada == 0:
                print(f"Época {epoca:4d} | error: {error_epoca:.4f}")

    def salida(self, x):
        # Valores de las neuronas de salida.
        with torch.no_grad():
            return self.modelo(self.a_tensor(x)).numpy()

    def predecir(self, x):
        # Clase predicha: la neurona de salida con mayor valor.
        return self.salida(x).argmax(axis=1)

    def exactitud(self, x, y):
        # Proporción de aciertos (y con las clases como enteros).
        return float((self.predecir(x) == np.asarray(y)).mean())

    def dibujar_error(self):
        import matplotlib.pyplot as plt
        plt.plot(range(1, len(self.historial) + 1), self.historial)
        plt.xlabel("Época")
        plt.ylabel("Error")
        plt.title("Evolución del error")
        plt.show()

    def __str__(self):
        texto = "NNet " + " → ".join(str(c) for c in self.capas) + "\n"
        texto += str(self.modelo) + "\n"
        texto += "Parámetros: " + str(sum(p.numel() for p in self.modelo.parameters()))
        return texto
