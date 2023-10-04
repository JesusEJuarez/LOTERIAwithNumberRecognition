# -*- coding: utf-8 -*-

#<================================ Primera parte ============================>#
#<=== Aquí manipulamos las imagenes

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
plt.close('all')
def evalPerceptron(p,w,b):
    a = w@p + b
    for k in range(10):
        if a[k]<0:
            a[k]=0
        else:
            a[k]=1
    return a
def entrenaPerceptron(P,T):
    b = np.random.rand(10,1)*2-1 
    a=np.zeros((10,1))
    wt= np.random.rand(10,10)*2-1   

    Epoca=200000
    for j in range(Epoca):
        for i in range(P.shape[1]):
            x = P[:,i]
            x= x[:,np.newaxis]
            # print(x)
            t = T[:,i]
            t= t[:,np.newaxis]
            # print(t)
            #print(x.shape)
            n=wt@x+b
            for k in range(10):
                if n[k]<0:
                    a[k]=0
                else:
                    a[k]=1
            e= t-a
            #print('ok')
            #print(e.shape)
            wt=wt+e*x.T
            b=b+e   
    return wt,b
# Loading 
direccion = "./entrenamiento"

lista_archivos = os.listdir(direccion)
lista_archivos = lista_archivos[:30] #tomar los primeros 10 
lista_archivos.sort()
listado=[]
num_images=0

k = 0
base_datos01= np.zeros((30*30, len(lista_archivos)))
for ii in lista_archivos:
    
    listado.append("./entrenamiento/"+ii)
    ima =  io.imread(listado[-1])
    ima = (transform.resize(ima,[30,30])>0.3).astype(int) #cambiar tamaño a 25x25
    # plt.figure(k+1)
    # plt.imshow(ima)
    base_datos01[:,k] = np.ravel(ima)  #convierte la matriz en un vector
    k += 1

# plt.close('all')
# plt.figure(0)
# plt.imshow(np.reshape(base_datos01[:,9],[25,25]), cmap = 'gray')
#se toma el 9no objeto y se cambia tamaño
# plt.show()

#<================================ Segunda parte ============================>#
#<=== Aquí entrenamos el backpropagation
#<=== Arquitectura neuronal con 64 neuronas de entrada y 3 neuronas de salida

w1 = np.random.rand(10, 900) * 2 - 1
b1 = np.random.rand(10, 1)   * 2 - 1
# plt.figure(1)
# plt.imshow(np.reshape(w1[0,:],[25,25]), cmap='gray')

w2 = np.random.rand(900, 10) * 2 - 1
b2 = np.random.rand(900, 1) * 2 - 1
target=base_datos01
alpha = 0.001
aprendizaje=[]
for j in range (200000):
    
    suma=0
    for i in range (target.shape[1]):
        a0 = base_datos01[:, i]
        a0 = a0[:, np.newaxis]
        n1 = w1@(a0) + b1
        a1 = 1 / (1 + np.exp(-n1))
        n2 = w2@(a1) + b2
        a2 = n2
        t  = target[:, i]
        t  = t[:, np.newaxis]
        e  = t - a2
        s2 = (-2)*1*e
        s1 = (np.diagflat((1-a1)*a1).dot(w2.T)).dot(s2)
        w2 = w2 - (alpha*s2*a1.T)
        b2 = b2 - (alpha*s2)
        w1 = w1 - (alpha*s1*a0.T)
        b1 = b1 - (alpha*s1)
        et=np.sum((np.sqrt(e**2))/900) #suma de errores cuadrados
        suma=et+suma #sumar los errores de cada numero
    aprendizaje.append(suma)

plt.figure()
plt.plot(aprendizaje)
#<================================ Tercera parte ============================>#
#<=== Se comprueba el entrenamiento el backpropagation

# cargar una base de datos distinta a la del entrenamiento
direccion = "./entrenamiento"

lista_archivos = os.listdir(direccion)
lista_archivos = lista_archivos[:30]
lista_archivos.sort()
listado=[]
num_images=0

k = 0
base_datos02 = np.zeros((30*30, len(lista_archivos)))
for ii in lista_archivos:
    listado.append("./entrenamiento/"+ii)
    ima =  io.imread(listado[-1]) 
    ima = (transform.resize(ima,[30,30])>0.3).astype(int)
    base_datos02[:,k] = np.ravel(ima)
    k += 1

salida = []
for i in range (target.shape[1]):
    a0 = base_datos02[:, i]
    a0 = a0[:, np.newaxis]
    n1 = w1@(a0) + b1
    a1 = 1 / (1 + np.exp(-n1))
    n2 = w2@(a1) + b2
    salida.append(n1)

# Import libraries
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

salida1 = np.array(salida)
for ii in range(10):
    ax.scatter( salida1[ii][0][0], salida1[ii][1][0], salida1[ii][2][0] )
    ax.text(    salida1[ii][0][0], salida1[ii][1][0], salida1[ii][2][0], ii )
    
 
    
 
T = np.zeros((10,30))
T[0,0] = 1 # cero
T[0,1] = 1 # cero
T[0,2] = 1 # cero


T[1,3] = 1 #uno
T[1,4] = 1 #uno
T[1,5] = 1 #uno


T[2,6] = 1 #dos
T[2,7] = 1 #dos
T[2,8] = 1 #dos


T[3,9] = 1 #tres
T[3,10] = 1 #tres
T[3,11] = 1 #tres


T[4,12] = 1 #cuatro
T[4,13] = 1 #cuatro
T[4,14] = 1 #cuatro


T[5,15] = 1 #cinco
T[5,16] = 1 #cinco
T[5,17] = 1 #cinco


T[6,18] = 1 #seis
T[6,19] = 1 #seis
T[6,20] = 1 #seis


T[7,21] = 1 #siete
T[7,22] = 1 #siete
T[7,23] = 1 #siete

T[8,24] = 1 #ocho
T[8,25]= 1 #ocho
T[8,26] = 1 #ocho


T[9,27] = 1 #nueve
T[9,28] = 1 #nueve
T[9,29] = 1 #nueve


P = np.array((salida[0],salida[1],salida[2],salida[3],salida[4],salida[5],salida[6],salida[7],salida[8],salida[9],salida[10],salida[11],salida[12],salida[13],salida[14],salida[15],salida[16],salida[17],salida[18],salida[19],salida[20],salida[21],salida[22],salida[23],salida[24],salida[25],salida[26],salida[27],salida[28],salida[29]))
P = P[:,:,0]*100
P  = P.T

wt,b = entrenaPerceptron(P, T)

for i in range(30):
    x = P[:,i]
    x= x[:,np.newaxis]
    a = evalPerceptron(x,wt,b)
    print('Numero:')
    print(i)
    print('\n Representación')
    print(a)
