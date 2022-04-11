import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings # Evita problemas
warnings.filterwarnings('ignore') # Evita problemas x2

data = pd.read_csv('iris.csv') # Apertura de csv
print (data.head()) # Impresión de cabecera de csv

data = data.loc[:, ['sepal length', 'sepal width', 'petal length', 'petal width']] # Lectura de las columnas para corroborar
print (data.head(10))

X = data.values # Asignación de valores del dataset a X y conversión a array numpy
X[:3]

sns.scatterplot (X[:,0], X[:, 1]) # Gráfico de dispersión

plt.xlabel('sepal length') # Etiqueta inferior
plt.ylabel('sepal width') # Etiqueta lateral izquierda
plt.title("Contenidos del Dataset Iris, Sépalos") # Título en márgen superior
plt.show() # Función de graficación

sns.scatterplot (X[:, 2], X[:, 3])

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title("Contenidos del Dataset Iris, Pétalos")
plt.show() # Impresión de la dispersión

k = int(input("\n-> Indique el número de grupos (centroides) a crear (k): "))
print("\n-> Número máximo de iteraciones: ", len(data)-2)

i = int(input("\n-> Indique el número de iteraciones (i): "))

r = int(input("\n-> Indique el número de corridas (r): "))

# Calcula WCSS (Within-Cluster Square Sum, Suma de Cuadrados dentro del cluster)
def calcula_wcss(X, centroides, cluster):
    sum = 0
    for i, val in enumerate(X):
        sum += np.sqrt((centroides[int(cluster[i]), 0]-val[0])**2 + (centroides[int(cluster[i]), 1]-val[1])**2 + (centroides[int(cluster[i]), 2]-val[2])**2 + (centroides[int(cluster[i]), 3]-val[3])**2)
    
    return sum

print ("\nLongitud: " + str(len(X)))

# Implementación de K-Means
def kmeans(X, k):
    diferencia = 1
    cluster = np.zeros(X.shape[0])
    
    # Selecciona k centroides al azar
    indices_azar = np.random.choice(len(X), size = k, replace = False, p = None)
    centroides = X[indices_azar, :]
    print ("\nCentroides azar: " + str(centroides))
    
    while diferencia:
        # Para cada observación
        for i, row in enumerate(X):
            dist_min = float('inf')
            
            # Distancia del punto desde todos los centroides
            for indice, centroide in enumerate(centroides):
                d = np.sqrt((centroide[0]-row[0])**2 + (centroide[1]-row[1])**2 + (centroide[2]-row[2])**2 + (centroide[3]-row[3])**2)
                # Almacena el centroide más cercano
                if dist_min > d:
                    dist_min = d
                    cluster[i] = indice
        
        nuevos_centroides = pd.DataFrame(X).groupby(by=cluster).mean().values
        print ("\nNuevos Centroides: " + str(nuevos_centroides))
        
        # Si los centroides son iguales entonces abandona
        if np.count_nonzero(centroides-nuevos_centroides) == 0:
            diferencia = 0
        else:
            centroides = nuevos_centroides
    
    return centroides, cluster

# Encuentra el valor de K usando el método Elbow (codo)
_lista = []
for k in range(1, 4):
    centroides, cluster = kmeans(X, k)

    # WCSS (Within-Cluster Sum of Squares)
    calcula = calcula_wcss(X, centroides, cluster)
    _lista.append(calcula)

    sns.scatterplot(X[:,0], X[:, 1], X[:, 2], X[:, 3], hue=cluster)
    sns.scatterplot(centroides[:,0], centroides[:, 1], centroides[:, 2], centroides[:, 3], s=100, color='y')

    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.show()