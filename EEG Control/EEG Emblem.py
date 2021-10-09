# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:40:26 2020

@author: Marco
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

global signal,C
#%% Archivos y Modelos
model_load = load('EEG/modeleeg.joblib')

Xfiltered=model_load['Xfiltered']
y = model_load['y']
model=model_load['model']

#%%Test
plt.plot(Xfiltered[y==4][np.random.randint(0,Xfiltered[y==4].shape[0])])
#%% HOTKEYS
Arriba = [ #Flecha Arriba
    {keyboard.KeyCode(char='w')},
    {keyboard.KeyCode(char='W')}
]

Abajo = [ #Flecha Abajo
    {keyboard.KeyCode(char='s')},
    {keyboard.KeyCode(char='S')}
]

Izq = [ #Flecha Izquierda
    {keyboard.KeyCode(char='a')},
    {keyboard.KeyCode(char='A')}
]

Der = [ #Flecha Derecha
    {keyboard.KeyCode(char='d')},
    {keyboard.KeyCode(char='D')}
]

Breaker = [ #Break
    {keyboard.KeyCode(char='q')},
    {keyboard.KeyCode(char='Q')}
]
# The currently active modifiers
current = set()

def sArriba():
    global signal
    global C
    signal=Xfiltered[y==4][np.random.randint(0,Xfiltered[y==4].shape[0])]
    C=1
    print("Comando: Arriba")
    
def sAbajo():
    global signal
    global C
    signal=Xfiltered[y==3][np.random.randint(0,Xfiltered[y==3].shape[0])]
    C=1
    print("Comando: Abajo")

def sDerecha():
    global signal
    global C
    signal=Xfiltered[y==2][np.random.randint(0,Xfiltered[y==2].shape[0])]
    C=1
    print("Comando: Derecha")
    
def sIzquierda():
    global signal
    global C
    signal=Xfiltered[y==1][np.random.randint(0,Xfiltered[y==1].shape[0])]
    C=1
    print("Comando: Izquerda")

def sBREAKER():
    global signal
    signal=np.array([])

def on_press(key):
    if any([key in COMBO for COMBO in Arriba]):
        current.add(key)
        if any(all(k in current for k in COMBO) for COMBO in Arriba):
            sArriba()
    elif any([key in COMBO for COMBO in Abajo]):
        current.add(key)
        if any(all(k in current for k in COMBO) for COMBO in Abajo):
            sAbajo()
    elif any([key in COMBO for COMBO in Der]):
        current.add(key)
        if any(all(k in current for k in COMBO) for COMBO in Der):
            sDerecha()
    elif any([key in COMBO for COMBO in Izq]):
        current.add(key)
        if any(all(k in current for k in COMBO) for COMBO in Izq):
            sIzquierda()
    elif any([key in COMBO for COMBO in Breaker]):
        current.add(key)
        if any(all(k in current for k in COMBO) for COMBO in Breaker):
            sBREAKER()


def on_release(key):
    if any([key in COMBO for COMBO in Arriba]):
        current.remove(key)

    elif any([key in COMBO for COMBO in Abajo]):
        current.remove(key)

    elif any([key in COMBO for COMBO in Der]):
        current.remove(key)

    elif any([key in COMBO for COMBO in Izq]):
        current.remove(key)

    elif any([key in COMBO for COMBO in Breaker]):
        current.remove(key)
        


#%% Main Code
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

C=0

signal=Xfiltered[y==4][np.random.randint(0,Xfiltered[y==4].shape[0])]
Tecla = keyboard.Controller()

while np.any(signal):
    if(C==1):
        prediction = int(model.predict(signal.reshape(1,-1)))
        C=0
        if(prediction==1):
            Tecla.press(keyboard.Key.left)
            time.sleep(0.05)
            Tecla.release(keyboard.Key.left)
            print("Resultado: Izquierda")
        elif(prediction==2):
            Tecla.press(keyboard.Key.right)
            time.sleep(0.05)
            Tecla.release(keyboard.Key.right)
            print("Resultado: Derecha")
        elif(prediction==3):
            Tecla.press(keyboard.Key.down)
            time.sleep(0.05)
            Tecla.release(keyboard.Key.down)
            print("Resultado: Abajo")
        elif(prediction==4):
            Tecla.press(keyboard.Key.up)
            time.sleep(0.05)
            Tecla.release(keyboard.Key.up)
            print("Resultado: Arriba")
listener.stop()
#%%
