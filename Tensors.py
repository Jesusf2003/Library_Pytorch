import torch as pt
import numpy as np

print("Versión de Pytorch: "+ pt.__version__)
print("Versión de Numpy: "+ np.__version__)
print()

data = [
    [1, 2, 3],
    [4, 5, 6],
    [4, 5, 6]
    ]

x = pt.zeros(3, 3)
y = pt.ones(3, 3)

print("Resultado: ", x + y)

'''
Tensores

    Son parecidos a arrays de NumPy que pueden
    usarse en un GPU para acelerar el cómputo.
'''

# Pytorch - Tensores

# Primera columna
print(x[1])
print()

# Trozo de la matriz
print(x[:-1, 1:])
print()

# Muestra la forma
print(x.shape)
print()

# Añadir dimensiones
print(x.view(1, 3, 3).shape)
print()

# Añadir valores restantes a una dimensión
print(x.view(-1).shape)
print()

# Pytorch - Autogradiantes

'''
Auto-gradiantes

    Provee una manera automática para identificar
    todas las operaciones sobre tensores
'''
# Crear un tensor y configurar requerimiento de gradiantes
x2 = pt.ones(2, 2, requires_grad=True)
y2 = x2 + 2
p = x2 + y2

'''
Función grad_in

    Da información sobre la operación anterior.
'''

# Mostrar atributo grad_fn
print(y.grad_fn)
print()

# Algunas operaciones en la variable y
z = y * y * 3 # Multiplicación
out = z.mean() # Promedio de todas las celdas

print(z, out)
print()

# Creamos tensor con entradas aleatorias
a = pt.randn(2, 2)

# Realizamos operaciones sobre tensor
a = ((a * 3) / (a - 1))

print(a.requires_grad)
print()

# Cambiamos el atributo requires_grad
a.requires_grad_(True)

print(a.requires_grad)
print()

# Otra operación
b = (a * a).sum()

print(b.grad_fn)
print()