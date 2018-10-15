# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 18:28:29 2018

@author: tanma
"""


import numpy as np
import pandas as pd

data = pd.read_csv('drake-songs.csv')

import re
text = ''

for index, row in data['lyrics'].iteritems():
    cleaned = str(row).lower().replace(' ', '\n').replace('|-|','\n')
    text = text + " ".join(re.findall(r"[a-z']+", cleaned))
    
tokens = re.findall(r"[a-z'\s]", text)

chars = sorted(list(set(tokens)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 1
sentences = []
next_char = []
for i in range(0, len(text)-maxlen, step):
  sentences.append(text[i:i+maxlen])
  next_char.append(text[i+maxlen])
  
x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
y = np.zeros((len(sentences), len(chars)))
  
for i, sentence in enumerate(sentences):
  for j, char in enumerate(sentence):
    x[i, j, char_indices[char]] = 1
  y[i, char_indices[next_char[i]]] = 1

from keras.models import Sequential
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Bidirectional
from keras.optimizers import RMSprop
from keras.layers import LSTM

model = Sequential()
model.add(Bidirectional(LSTM(128, input_shape=(maxlen,len(chars)), return_sequences = True)))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer =RMSprop(lr=0.01))

filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath,monitor = "loss", verbose = 1,save_best_only = True,mode = "min")
callbacks_list = [checkpoint]

model.fit(x, y, batch_size = 1280, epochs = 10,callbacks = callbacks_list)  