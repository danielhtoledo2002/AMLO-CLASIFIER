# -*- coding: utf-8 -*-


import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import json


#load model
new_model =  keras.models.load_model('DeepLearningModels/RNN_model.h5', compile=False)
new_model.compile()

#load tokenizer
with open('DeepLearningModels/tokenizer_rnn.json') as f:
    data = json.load(f)
    new_tokenizer = tokenizer_from_json(data)

#open sequence len
f = open("DeepLearningModels/seq_len.txt", "r")

#save sequence len in a variable
line = f.readlines()
for l in line:
  if l != '':
    seq_len = l
seq_len = int(seq_len)

#generate phtase
text_len = 20

def generate_text(seed_text, n_lines):
  text2 = ''
  for i in range(n_lines):
    text = []
    for _ in range(text_len):
      encoded = new_tokenizer.texts_to_sequences([seed_text])
      encoded = pad_sequences(encoded, maxlen=seq_len, padding="pre")
      y_pred = np.argmax(new_model.predict(encoded, verbose =0), axis=-1)
      predicted_word = ""
      for word, index in new_tokenizer.word_index.items():
        if index == y_pred:
          predicted_word = word
          break
      seed_text = seed_text + " " + predicted_word
      text.append(predicted_word)
    seed_text = text[-1]
    text = " ".join(text)
    text2 = text2 + text
  return text2

generate_text('opinion de la corrupcion', 3)