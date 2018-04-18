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
        #np.random.seed(16)
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

    def predict(self, X, saida):
        print X.shape[0]
        predicts = np.zeros(X.shape[0])

        print 'Resultado |', 'Valor |', 'Valor esperado'
        print '------------------------------------'

        for n in xrange(X.shape[0]):
            xt = np.array([-1, X[n, 0], X[n, 1]]).reshape(-1, 1)
            y = np.dot(self.W, xt)[0][0] # Combinador linear

            if (y < 0):
                print "setosa |", y, '|', saida[n]
            else:
                print "vesicolor |", y, '|', saida[n]

        

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

### Plotando os dados de treinamento
### Pontos vermelhos ----> Setosa (-1)
### Pontos azuis ----> Versicolor (1)
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.figure(figsize=(7,5))
plt.scatter(X_train[:,0], X_train[:,1], c=d, cmap=cm_bright)
plt.scatter(None, None, color = 'r', label='Setosa')
plt.scatter(None, None, color = 'b', label='Versicolor')
plt.legend()
plt.title('Dados de Treinamento')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()

########################################

adaline = Adaline()
adaline.train(X_train, d)

adaline.predict(X_test, y)

"""
######### Plotando os limites de decisão (depois do treinamento) #############  

# Set x_min, x_max, y_min, y_max
x_min, x_max = X_test[:, 0].min() - 2., X_test[:, 0].max() + .5
y_min, y_max = X_test[:, 0].min() - 2, X_test[:, 0].max()

# Step size in the mesh
h = 0.001
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = adaline.predict(np.c_[xx.ravel(), yy.ravel()])

# Crete color for training point and test point
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

# Plot the decision boundary and scatter labels
plt.figure(figsize=(7,5))
plt.contourf(xx, yy, Z, cmap=cm, alpha=.9)
plt.scatter(X_test[:,0], X_test[:,1], c=y, cmap=cm_bright)
plt.scatter(None, None, color = 'r', label='Setosa')
plt.scatter(None, None, color = 'b', label='Versicolor')
plt.legend()
plt.xlim([x_min + 1.0, x_max])
plt.ylim([y_min + 0.5, y_max - 3.0])
plt.title('The Decision Boundary of Adaline after training')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()

R = adaline.predict(X_test)

#print iris_datasets.feature_names
#print iris_datasets.target_names
#print iris_datasets.target
#print iris_datasets"""