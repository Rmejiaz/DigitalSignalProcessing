# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:40:26 2020

@author: Marco
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard
import pyaudio
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from SpeechFunctions import *
from python_speech_features import mfcc

#%% Archivos y Modelos
model_load = load('speech_model7.joblib')
#%%
labels = model_load['labels']
model=model_load['best_model_7']
#%% HOTKEYS
COMBINATIONS = [
    {keyboard.KeyCode(char='q')},
    {keyboard.KeyCode(char='Q')}
]
Fast = [
    {keyboard.KeyCode(char='e')},
    {keyboard.KeyCode(char='e')}
]
# The currently active modifiers
current = set()

def on_press(key):
    if any([key in COMBO for COMBO in COMBINATIONS]):
        current.add(key)
        if any(all(k in current for k in COMBO) for COMBO in COMBINATIONS):
            execute()
    if any([key in COMBO for COMBO in Fast]):
        current.add(key)
        if any(all(k in current for k in COMBO) for COMBO in Fast):
            Faster()

def on_release(key):
    if any([key in COMBO for COMBO in COMBINATIONS]):
        current.remove(key)
    if any([key in COMBO for COMBO in Fast]):
        current.remove(key)
        
def execute():
    global k
    k=False
    
def Faster():
    global i
    i+=1
    if i==5:
        i=1
    print("speed set to: "+str(i))
#%% Main Code
listener = keyboard.Listener(on_press=on_press, on_release=on_release)

listener.start()

Tecla = keyboard.Controller()

global k,i
i=1
k=True
p = pyaudio.PyAudio()

RATE    = 48000
CHUNK   = 2**13

player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, 
frames_per_buffer=CHUNK)
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

while(k):
    signal=np.array([])
    Q=False
    while(True):
        Read=np.frombuffer(stream.read(CHUNK),dtype=np.int16)
        # player.write(Read,CHUNK)
        if (Q and np.all(np.abs(Read)<600)):
            break
        elif (Q or (np.linalg.norm(Read)/CHUNK)>7.5):
            signal= np.append(signal,Read)
            Q=True
    if (signal.shape[0]<96000):
        signal = np.pad(signal,(0,2*RATE-signal.shape[0]),'constant',constant_values=(0,0)) 
        signal=np.roll(signal,33000)
        plt.plot(signal)
        plt.pause(0.00001)
        prediction = int(model.predict(signal.reshape(1,-1)))
        print( f"Resultado: {labels[prediction]} - {prediction}")
        if(prediction==2):
            for j in range(0,i):
                Tecla.press(keyboard.Key.left)
                time.sleep(0.05)
                Tecla.release(keyboard.Key.left)
                time.sleep(0.05)
        elif(prediction==3):
            for j in range(0,i):
                Tecla.press(keyboard.Key.right)
                time.sleep(0.05)
                Tecla.release(keyboard.Key.right)
                time.sleep(0.05)
        elif(prediction==1):
            for j in range(0,i):
                Tecla.press(keyboard.Key.down)
                time.sleep(0.05)
                Tecla.release(keyboard.Key.down)
                time.sleep(0.05)
        elif(prediction==0):
            for j in range(0,i):
                Tecla.press(keyboard.Key.up)
                time.sleep(0.05)
                Tecla.release(keyboard.Key.up)
                time.sleep(0.05)
        elif(prediction==5):
            Tecla.press('z')
            time.sleep(0.05)
            Tecla.release('z')
            time.sleep(0.05)
        elif(prediction==4):
            Tecla.press('x')
            time.sleep(0.05)
            Tecla.release('x')
            time.sleep(0.05)
stream.stop_stream()
stream.close()
p.terminate()
listener.stop()
#%%
dump(signal,'get_a_job.joblib')
#%%
X=np.copy(load('fuck.joblib'))