import torch as pt
import numpy as np

from sklearn.datasets import fetch_openml


# Modelos secuenciales
D_in, H, D_out = 784, 100, 10

model = pt.nn.Sequential(
     pt.nn.Linear(D_in, H),
     pt.nn.ReLU(),
     pt.nn.Linear(H, D_out),
)

# outputs = model(pt.randn(64, 784))
# print(outputs.shape)

model.to("cpu")


# Descarga de datos

'''
Dataset MNIST

    Una gran base de datos de dígitos escritos a mano
    usada comúnmente para entrenar varios sistemas de
    procesamiento de imágenes.
'''

mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist["data"], mnist["target"]
