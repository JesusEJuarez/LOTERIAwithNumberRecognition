# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:07:35 2022

@author: tetis
"""
from time import sleep
import numpy as np
from skimage import io, color, transform,  measure
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import random
from PIL import Image
from skimage.morphology import disk, erosion, dilation
from skimage.filters.rank import  median
import cv2

plt.close('all')

def limpieza(binaria):
    tamanio= np.shape(binaria)[1]
    ero = round(np.shape(binaria)[0]*tamanio*0.0000016)
    limpia = 1-binaria
    limpia = ndimage.binary_fill_holes(limpia).astype(int)
    limpia = erosion(limpia,disk(ero))
    limpia = ndimage.binary_fill_holes(limpia).astype(int)
    limpia = dilation(limpia, disk(ero*2))
    limpia = median(limpia)
    limpia = ndimage.binary_fill_holes(limpia).astype(int)
    limpia = 1-limpia
    return limpia
def binaria(num):
    binaria=((color.rgb2gray(num)> 0.35)).astype(int) 
    return binaria

def segmenta2(binaria):
    perfil = np.sum(binaria, axis = 1)
    maximo = np.max(perfil)
    a = True
    b =0
    while(a):
        while(perfil[b] >= (maximo-maximo/100)):
            b = b+1
        else:
            liminf = b
        while(perfil[b] <= maximo-maximo/100 ):
            limsup = b
            if(b>=(len(perfil)-3)):
                break
            b = b+1
        else:
            limsup = b
        binaria = binaria[liminf:limsup,:]
        if((limsup-liminf)<len(perfil)*0.01):
            b = limsup
        else:
            a = False
    return binaria 
def segmentaV(binaria,b):
    perfil = np.sum(binaria, axis = 0)
    maximo = np.max(perfil)
    a = True
    while(a):
        while(perfil[b] >= (maximo-maximo/100)):
            b = b+1
        else:
            liminf = b
        while(perfil[b] <= maximo-maximo/100 ):
            limsup = b
            if(b>=(len(perfil)-3)):
                break
            b = b+1
        else:
            limsup = b
        binaria = binaria[:,liminf:limsup]
        if((limsup-liminf)<len(perfil)*0.01):
            b = limsup
        else:
            a = False
    return [binaria,limsup] 
def tableroAletorio():
    lista = range(1,55)
    cartas = random.sample(lista,16)
    cartas = [cartas[0:4],cartas[4:8],cartas[8:12],cartas[12:16]]
    return cartas

def tableroAletorio2():
    lista = range(1,55)
    cartas = random.sample(lista,16)
    num = [28,5,30,54,42,3,51,21,24,18,23,50,20,53,25,29]
    cartas = [num[0:4],num[4:8],num[8:12],num[12:16]]
    return cartas

                
def crearTablero(cartas,jugador,pasadas):
    fig, axs = plt.subplots(4, 4)
    #gs = fig.add_gridspec(4, 4, hspace=0, wspace=0)
    fig.suptitle('Carta   '+jugador)
    for i in range(0,4):
        for j in range(0,4):
            axs[j, i].imshow(io.imread("Cartas/" + str(cartas[j][i])+".jpg"))
            if pasadas[j][i] == 1:
                angulo = np.linspace(0, 2*np.pi, 100+1)
                x = 100 * np.cos(angulo) + 620
                y = 100 * np.sin(angulo) + 877
                axs[j, i].plot(x, y, color="fuchsia", markersize=1,linewidth=10)
            axs[j, i].axis('off')
            axs[j, i].set_xticklabels([])
            axs[j, i].set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.pause(1)
            
def compreubaCartas(cartas, actual,pasadas):
    for i in range(0,4):
        for j in range(0,4):
            if cartas[j][i] == actual:
                pasadas[j][i] = 1
                
    return pasadas
def esGanador(pasadas):
    pasadas = np.array(pasadas)
    diagonal1 =[]
    diagonal2 = []
    for i in range(4):
        suma = sum(pasadas[:,i])
        if suma == 4:
            return True
        suma = sum(pasadas[i,:])
        if suma == 4:
            return True
        diagonal1.append(pasadas[i,i])
        diagonal2.append(pasadas[3-i,i])
    if sum(diagonal1) == 4 or sum(diagonal2) == 4:
        return True
    else:
        return False


def tomaFoto(liy,lsy,lix,lsx):
    cap = cv2.VideoCapture(0)
    leido, Imagr = cap.read()
    # plt.figure(0)
    # plt.imshow(Imagr)
    bina=binaria(Imagr)
    bina=limpieza(bina)
    rec=bina[liy:lsy,lix:lsx]
   # plt.figure(1)
    #plt.imshow(rec)
    tiporec=(rec)*255
    tiporec=255-tiporec
    #plt.figure(2)
    #plt.imshow(tiporec)
    labels = measure.label(tiporec, connectivity=2, background=0)
    print(np.max(labels))
    if np.max(labels)>=2:
        print('dos elementos')
        rec1,b=segmentaV(rec,0)
        rec2,a=segmentaV(rec,b)
    elif np.max(labels)==1:
        print('1 elementos')
        rec1=rec
        rec1,b=segmentaV(rec1,0)
        rec2 =[]
    return [rec1,rec2]




def evalPerceptron(p,w,b):
    a = w@p + b
    for k in range(10):
        if a[k]<0:
            a[k]=0
        else:
            a[k]=1
    return a

def evalBack(a0,w1,b1):
    n1 = w1@(a0) + b1
    
    return n1

def identificaNum(n):
    for i in range(10):
        if (n[i] == 1):
            if(i == 0):
                a=0
            elif(i == 1):
                a=1
            elif(i == 2):
                a=2##
            elif(i == 3):
                a=3
            elif(i == 4):
                a=4
            elif(i == 5):
                a=5 ##
            elif(i == 6):
                a=6
            elif(i == 7):
                a=7
            elif(i == 8):
                a=8
            elif(i == 9):
                a=9##
            break
        else:
            a =1
    return a

def numActual(rec1,rec2):
    wt = np.load('./variables/wt.npy')
    w1 = np.load('./variables/w1.npy')
    b = np.load('./variables/b.npy')
    b1 = np.load('./variables/b1.npy')
    print('okokoko')
    if type(rec1) == type(rec2):
        print('okokoko')
        num1 = (transform.resize(rec1,[30,30])>0.999E-7).astype(int)
        num2 = (transform.resize(rec2,[30,30])>0.999E-7).astype(int)
        #num1 = transform.resize(rec1,[30,30])
        #num2 = transform.resize(rec2,[30,30])
        num1 = np.ravel(num1)
        num2 = np.ravel(num2)
        num1 = num1[:, np.newaxis]
        num2 = num2[:, np.newaxis]
        num1 = evalBack(num1,w1,b1)*100
        num2 = evalBack(num2,w1,b1)*100
        num1 = evalPerceptron(num1,wt,b)
        num2 = evalPerceptron(num2,wt,b)
        identificador = str(identificaNum(num1)) + str(identificaNum(num2))
        salida = int(identificador)
    else:
        num1 =(transform.resize(rec1,[30,30])>0.999E-7).astype(int)
        #num1 = transform.resize(rec1,[30,30])
        plt.imshow(num1)
        num1 = np.ravel(num1)
        num1 = num1[:, np.newaxis]
        num1 = evalBack(num1,w1,b1)*100
        num1 = evalPerceptron(num1,wt,b)
        identificador = identificaNum(num1)
        salida = int(identificador)
    return salida 


#    inicio de juego 
pasadasJugador = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
pasadasCompu = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

tableroJugador = tableroAletorio()
tableroCompu = tableroAletorio2()
ganaJugador = False
ganaCompu = False
a = True
while(a):
    plt.close('all')
    crearTablero(tableroJugador, 'jugador', pasadasJugador)
    crearTablero(tableroCompu, 'Compu', pasadasCompu)
    sleep(2)
    rec1,rec2 = tomaFoto(43,156,252,400)
    
    rec1=segmenta2(rec1)
    if type(rec1) == type(rec2):
        rec2=segmenta2(rec2)
        
    rec1=rec1*255
    rec2=rec2*255
    
    actual = numActual(rec1, rec2)
    pasadasJugador = compreubaCartas(tableroJugador, actual,pasadasJugador)
    pasadasCompu = compreubaCartas(tableroCompu, actual,pasadasCompu)
    
    ganaJugador = esGanador(pasadasJugador)
    ganaCompu = esGanador(pasadasCompu)
    
    if ganaJugador and ganaCompu:
        print('Es un empate')
        a = False
    else:
        if ganaJugador:
            marco = '#'
            marco = marco*30
            print(marco + '\n \n La Felicidades usted ha ganado \n \n'+ marco)
            a = False
        elif ganaCompu:
            marco = '#'
            marco = marco*55
            print(marco + '\n \n Hasta la computadora tiene más surte, ha perdido T.T \n \n'+ marco)
            a = False
    



