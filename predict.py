import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from nlp import preprocess
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, concatenate
from tensorflow.keras.models import Model

sys.path.append(Path(os.path.join(os.path.abspath(''), '../')).resolve().as_posix())

tokenizer_path = Path('../datasets/tokenizer.pickle').resolve()
with tokenizer_path.open('rb') as file:
    tokenizer = pickle.load(file)

input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
num_classes = 4
embedding_dim = 500
input_length = 100
lstm_units = 128
lstm_dropout = 0.1
recurrent_dropout = 0.1
spatial_dropout = 0.2
filters = 64
kernel_size = 3

input_layer = Input(shape=(input_length,))
output_layer = Embedding(
  input_dim=input_dim,
  output_dim=embedding_dim,
  input_shape=(input_length,)
)(input_layer)

output_layer = SpatialDropout1D(spatial_dropout)(output_layer)

output_layer = Bidirectional(
    LSTM(lstm_units, return_sequences=True,
         dropout=lstm_dropout, recurrent_dropout=recurrent_dropout)
)(output_layer)
output_layer = Conv1D(filters, kernel_size=kernel_size, padding='valid',
                      kernel_initializer='glorot_uniform')(output_layer)

avg_pool = GlobalAveragePooling1D()(output_layer)
max_pool = GlobalMaxPooling1D()(output_layer)
output_layer = concatenate([avg_pool, max_pool])

output_layer = Dense(num_classes, activation='softmax')(output_layer)

model = Model(input_layer, output_layer)

model_weights_path = Path('../model_weights.h5').resolve()
model.load_weights(model_weights_path.as_posix())

# Loading unlabeled datasets.

data_path = Path('../datasets/full_dataset.csv').resolve() # this dataset still dont exist
data = pd.read_csv(data_path)
data.head()
data.replace('', np.nan, inplace=True)
data.dropna(inplace=True)
data.shape
data["full_text"].astype(str)
data.head()

encoder_path = Path('../emotion_recognition/encoder.pickle').resolve()
with encoder_path.open('rb') as file:
    encoder = pickle.load(file)

cleaned_data = preprocess(data.full_text)
sequences = [text.split() for text in cleaned_data]
list_tokenized = tokenizer.texts_to_sequences(sequences)
x_data = pad_sequences(list_tokenized, maxlen=100)

y_pred = model.predict(x_data)

for index, value in enumerate(np.sum(y_pred, axis=0) / len(y_pred)):
    print(encoder.classes_[index] + ": " + str(value))

y_pred_argmax = y_pred.argmax(axis=1)
data_len = len(y_pred_argmax)
for index, value in enumerate(np.unique(y_pred_argmax)):
    print(encoder.classes_[index] + ": " + str(len(y_pred_argmax[y_pred_argmax == value]) / data_len))

y_pred[30:41].argmax(axis=1)

data.full_text.iloc[36]
