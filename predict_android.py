# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:32:47 2018

@author: tanma
"""

# Importing Modules
import numpy as np
import pandas as pd
import random
import re
import os
from keras.models import load_model


# Importing Data
data = pd.read_csv('drake-songs.csv')

text = ''

# Cleaning Data
for index, row in data['lyrics'].iteritems():
    cleaned = str(row).lower().replace(' ', '\n').replace('|-|','\n')
    text = text + " ".join(re.findall(r"[a-z']+", cleaned))
    
tokens = re.findall(r"[a-z'\s]", text)

chars = sorted(list(set(tokens)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

vocab_size = len(chars)

# Vectorizing
maxlen = 40
step = 1
sentences = []
next_char = []
for i in range(0, len(text)-maxlen, step):
  sentences.append(text[i:i+maxlen])
  next_char.append(text[i+maxlen])

# One Hot Encoding  
x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)))
  
for i, sentence in enumerate(sentences):
  for j, char in enumerate(sentence):
    x[i, j, char_indices[char]] = 1
  y[i, char_indices[next_char[i]]] = 1

# Loading the Saved Trained Model
model = load_model("model.h5")  
  
generated = ''
start_index = random.randint(0,len(text)-maxlen-1)
sent = text[start_index:start_index+maxlen]
generated += sent
for i in range(200):
    x_sample = generated[i:i+maxlen]
    x = np.zeros((1,maxlen,vocab_size))
    for j in range(maxlen):
        x[0,j,char_indices[x_sample[j]]] = 1
    probs = model.predict(x)
    probs = np.reshape(probs,probs.shape[1])
    ix = np.random.choice(range(vocab_size),p=probs.ravel())
    generated += indices_char[ix]
   
print(generated)

#Pushing Data to Firebase
from google.cloud import firestore
path=r"C:\Users\tanma\Desktop\GitHub\lyrics_generator_using-LSTM\neural-net-lyric-generator-firebase-adminsdk-kwdi8-b2b16610ef.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=path
db=firestore.Client()
generations = db.collection(u'Generated-Data').document(u"Data")
generations.update({
    u'Start' : sent,
    u'Generated' : generated
            })