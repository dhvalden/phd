import json
import pandas as pd
import re
from pathlib import Path


data = []
with open("../datasets/bigemoji.json", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

del data

df.to_csv("../datasets/dataemo.csv", index=None)

relations_path = Path('../query_relations.json')
with relations_path.open('r') as file:
    relations = json.load(file)

emotion = 'anger'
queries = [key for key, value in relations.items() if value == emotion]

from emoji import demojize, emojize

data_emojis = df.text.apply(lambda x: re.findall(r':[a-z_]+:', demojize(x)))

emoji_dict = {}
for i, emojis in data_emojis.iteritems():
    for emoji in emojis:
        if emoji in emoji_dict:
            emoji_dict[emoji] += 1
        else:
            emoji_dict[emoji] = 1

for emoji, count in sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True):
    print(emojize(emoji) + '(' + emoji + '): ' + str(count))

data_hashtags = df.text.apply(lambda x: re.findall(r'#\S+', x))

hashtag_dict = {}
for i, hashtags in data_hashtags.iteritems():
    for hashtag in hashtags:
        if hashtag in hashtag_dict:
            hashtag_dict[hashtag] += 1
        else:
            hashtag_dict[hashtag] = 1

for hashtag, count in sorted(hashtag_dict.items(), key=lambda x: x[1], reverse=True):
    print(hashtag + ': ' + str(count))

import os
import sys

import json


from pathlib import Path

relations_path = Path('../query_relations.json').resolve()

with open('../query_relations.json', 'r') as file:
    relations = json.load(file)

data = pd.read_csv("../datasets/training.1600000.processed.noemoticon.csv", encoding="latin-1", names=["label", "id", "date", "query", "user", "tweet"])

import re
from time import time
import nltk
from emoji import demojize

nltk.download('stopwords')

texts = data.tweet

start = time()
# Lowercasing
texts = texts.str.lower()

# Remove special chars
texts = texts.str.replace(r"(http|@)\S+", "")
texts = texts.apply(demojize)
texts = texts.str.replace(r"::", ": :")
texts = texts.str.replace(r"’", "'")
texts = texts.str.replace(r"[^a-z\':_]", " ")

# Remove repetitions
pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
texts = texts.str.replace(pattern, r"\1")

# Transform short negation form
texts = texts.str.replace(r"(can't|cannot)", 'can not')
texts = texts.str.replace(r"n't", ' not')

# Remove stop words
stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove('not')
stopwords.remove('nor')
stopwords.remove('no')
texts = texts.apply(
    lambda x: ' '.join([word for word in x.split() if word not in stopwords])
)

print("Time to clean up: {:.2f} sec".format(time() - start))

data.tweet = texts

num_words = 10000

import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=num_words, lower=True)
tokenizer.fit_on_texts(data.tweet)

with open('../datasets/tokenizer.pickle', 'wb') as file:
    pickle.dump(tokenizer, file)

from sklearn.model_selection import train_test_split

train = pd.DataFrame(columns=['label', 'tweet'])
validation = pd.DataFrame(columns=['label', 'tweet'])
for label in data.label.unique():
    label_data = data[data.label == label]
    train_data, validation_data = train_test_split(label_data, test_size=0.3)
    train = pd.concat([train, train_data])
    validation = pd.concat([validation, validation_data])

from tensorflow.keras.layers import Input, Embedding, GRU
from tensorflow.keras.layers import Dropout, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Dense
from tensorflow.keras.models import Sequential

input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
embedding_dim = 200
input_length = 100
gru_units = 128
gru_dropout = 0.1
recurrent_dropout = 0.1
dropout = 0.1

model = Sequential()
model.add(Embedding(
    input_dim=input_dim,
    output_dim=embedding_dim,
    input_shape=(input_length,)
))

model.add(Bidirectional(GRU(
    gru_units,
    return_sequences=True,
    dropout=gru_dropout,
    recurrent_dropout=recurrent_dropout
)))
model.add(GlobalMaxPooling1D())
model.add(Dense(32, activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

from tensorflow.keras.preprocessing.sequence import pad_sequences

train_sequences = [text.split() for text in train.tweet]
validation_sequences = [text.split() for text in validation.tweet]
list_tokenized_train = tokenizer.texts_to_sequences(train_sequences)
list_tokenized_validation = tokenizer.texts_to_sequences(validation_sequences)

x_train = pad_sequences(list_tokenized_train, maxlen=input_length)
x_validation = pad_sequences(list_tokenized_validation, maxlen=input_length)
y_train = train.label.replace(4, 1)
y_validation = validation.label.replace(4, 1)

batch_size = 128
epochs = 1

model.fit(
    x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_validation, y_validation),
)

## New section

with open('../query_relations.json','r') as file:
    relations = json.load(file)
import pickle

with open('../datasets/tokenizer.pickle','rb') as file:
    tokenizer = pickle.load(file)

    
from tensorflow.keras.layers import Input, Embedding, GRU
from tensorflow.keras.layers import Dropout, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Dense
from tensorflow.keras.models import Sequential

input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
embedding_dim = 200
input_length = 100
gru_units = 128
gru_dropout = 0.1
recurrent_dropout = 0.1
dropout = 0.1

model = Sequential()
model.add(Embedding(
    input_dim=input_dim,
    output_dim=embedding_dim,
    input_shape=(input_length,)
))

model.add(Bidirectional(GRU(
    gru_units,
    return_sequences=True,
    dropout=gru_dropout,
    recurrent_dropout=recurrent_dropout
)))
model.add(GlobalMaxPooling1D())
model.add(Dense(32, activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(1, activation='sigmoid'))

print(model.summary())

weights_path = Path('../datasets/model_weights.h5').resolve()
model.load_weights(weights_path.as_posix())

import os
import re
import pandas as pd
from tqdm import tqdm

emotion_data_dict = {}
query = "anger"
relations
emotion = relations[query]
file_data = pd.read_csv("../datasets/dataemo.csv")
file_data.head()
dict_data = emotion_data_dict[emotion] if emotion in emotion_data_dict else None
emotion_data_dict[emotion] = pd.concat([dict_data, file_data])
t.update()

relations
