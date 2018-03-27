# encoding: utf8

# Baseado em http://www.triszaska.com/2017/06/implement-adaline-in-python-to-classify-iris-data.html

import matplotlib.pyplot as plt # Para plotar graficos
from sklearn import datasets # Biblioteca sklearn para carregar os dados
import numpy as np # Array do Python
from matplotlib.colors import ListedColormap # Contém uma lista de cores para usar nos graficos

class Adaline(object):
    def __init__(self, eta=0.001, max_epoch=1000):
        self.eta = eta
        self.max_epoch = max_epoch
        self.Wini = 0
        self.W = None
        self.N = 0

    def train(self, X, d):
        self.N = d.size

        # Iniciar pesos sinápticos
        np.random.seed(16)
        self.Wini = np.random.uniform(-1, 1, X.shape[1] + 1).reshape(1, -1)
        W = self.Wini
        print W
        W1 = []
        W1.append(W)
        
        CP = 0
        SSE = np.zeros(self.max_epoch)
        E = np.zeros(self.N)

        for epoca in xrange(self.max_epoch):
            idc = np.random.permutation(self.N)
            CP = 0

            for n in xrange(self.N):
                xt = np.array([-1, X[idc[n], 0], X[idc[n], 1]]).reshape(-1, 1)
                y = np.dot(W, xt)[0][0] # Combinador linear

                e = d[idc[n]] - y

                W = W + (self.eta * e) * xt.T

                E[n] = 0.5 * e**2
            
            W1.append(W)
            SSE[epoca] = np.sum(E) / self.N

            if (epoca > 0):
                dSSE = abs(SSE[epoca] - SSE[epoca - 1])
                
                if (dSSE < 1e-6):
                    CP += 1
            
            if (CP > 3):
                break
        self.W = W

    def predict(self, X):
        for n in xrange(X.shape[0]):
            xt = np.array([-1, X[n, 0], X[n, 1]]).reshape(-1, 1)
            y = np.dot(self.W, xt)[0][0] # Combinador linear

            if (y < 0):
                print "setosa", y
            else:
                print "vesicolor", y

        

# Carregando conjunto de dados da Iris
iris_datasets = datasets.load_iris()

### Preparando o conjunto de dados de treinamento e o conjunto de dados de teste

X_train = []
X_test = []
d = []
y = []

cont = 0

for i in xrange(100):
    if (i >= 0 and i < 40): # Dados de treinamento referentes à setosa
        X_train.append(iris_datasets.data[i][2:])
        d.append(-1)
    elif (i >= 40 and i < 50): # Dados de teste referentes à setosa
        X_test.append(iris_datasets.data[i][2:])
        y.append(-1)
    elif (i >= 50 and i < 90): # Dados de treinamento referentes à vesicolor
        X_train.append(iris_datasets.data[i][2:])
        d.append(1)
    elif (i >= 90 and i < 100): # Dados de teste referentes à vesicolor
        X_test.append(iris_datasets.data[i][2:])
        y.append(1)

    #print 'Exemplo %d: Label %s, feature %s' % (i, iris_datasets.target[i], iris_datasets.data[i][2:])

# Convertendo as listas em Numpy
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
d = np.asarray(d)
y = np.asarray(y)

### Scatter plot iris data
### The red dots ----> Setosa (-1)
### The blue dots ----> Versicolor (1)
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.figure(figsize=(7,5))
plt.scatter(X_train[:,0], X_train[:,1], c=d, cmap=cm_bright)
plt.scatter(None, None, color = 'r', label='Setosa')
plt.scatter(None, None, color = 'b', label='Versicolor')
plt.legend()
plt.title('Visualize the data')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()

########################################

adaline = Adaline()
adaline.train(X_train, d)

adaline.predict(X_test)

#print iris_datasets.feature_names
#print iris_datasets.target_names
#print iris_datasets.target
#print iris_datasets