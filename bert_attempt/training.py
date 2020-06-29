import pandas as pd
import pickle
from bert_attempt.nlp.dataset import Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer


#####################

widedata = pd.read_json('bert_attempt/datasets/widelabeled.json')
widedata.head(100)

one_hot = MultiLabelBinarizer()

onehotlabels = one_hot.fit_transform(widedata['values'])

one_hot.classes_

widedf = pd.DataFrame(onehotlabels, columns=one_hot.classes_)

widedf['text'] = widedata['key']

widedf.to_csv('bert_attempt/datasets/widelabeled.csv')


###################

dataset_path = 'bert_attempt/datasets/widelabeled.csv'

labels_columns = ['♥', '♥️', '⚠', '⚠️', '✊', '✊🏽', '✌', '❌', '❤', '❤️', '🆘', '🌍',
                  '🌎', '🌏', '🎉', '👀', '👊', '👋', '👋🏽', '👍', '👎', '👏', '👏🏻', '👏🏼',
                  '👏🏽', '💀', '💔', '💙', '💚', '💜', '💥', '💩', '💪', '🔥', '😀', '😁', '😂',
                  '😅', '😆', '😉', '😊', '😍', '😎', '😠', '😡', '😢', '😭', '😱', '😳', '😴',
                  '🙄', '🙋', '🙋🏻', '🙋🏻\u200d♂', '🙋🏻\u200d♂️', '🙌', '🙏', '🙏🏻', '🚨',
                  '🤔', '🤡', '🤣', '🤦', '🤦\u200d♀', '🤦\u200d♀️', '🤦\u200d♂',
                  '🤦\u200d♂️', '🤪', '🤬', '🤮', '🤷', '🥴', '🥺', '🧐']

dataset = Dataset(dataset_path, label_col=labels_columns, text_col='text')
dataset.load()
dataset.preprocess_texts()

dataset.cleaned_data.columns

count = dataset.cleaned_data['text'].str.split().str.len()
data = dataset.cleaned_data[count > 1]
ia = data[data['label'] == 'This is not norma…'].index
ib = data[data['label'] == '!'].index
data = data.drop(ia)
data = data.drop(ib)

data.to_csv('bert_attempt/datasets/wide_preprocesed_data.csv')

##################

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
import tensorflow as tf

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

from emoji import demojize

mydata = pd.read_csv('bert_attempt/datasets/wide_preprocesed_data.csv', index_col=0)

text = mydata['text']
labels = mydata.drop('text', axis=1)

emojicolnames = []
for ele in labels.columns:
    emojicolnames.append(demojize(ele))
labels.columns = emojicolnames

labels.columns.duplicated()
labels.columns

# :heart_suit:
heart_suit = labels.iloc[:,0] + labels.iloc[:,1]
heart_suit.value_counts()
heart_suit.replace(2, 1, inplace=True)
heart_suit.value_counts()

# :warning:
warnings = labels.iloc[:,2] + labels.iloc[:,3]
warnings.value_counts()
warnings.replace(2, 1, inplace=True)
warnings.value_counts()


# :red_heart:
red_heart = labels.iloc[:,8] + labels.iloc[:,9]
red_heart.value_counts()
red_heart.replace(2, 1, inplace=True)
red_heart.value_counts()

# replacing

todrop = list(~labels.columns.duplicated())

labels.columns[todrop]

labels = labels.iloc[:,todrop]

labels = labels.assign(heart_suit=labels[':heart_suit:'])
labels = labels.assign(warnings=labels[':warning:'])
labels = labels.assign(red_heart=labels[':red_heart:'])
labels = labels.drop(columns=['heart_suit', 'warnings', 'red_heart'])
labels.shape
text.shape

text.to_csv('bert_attempt/datasets/text_column.csv', index=False)
labels.to_csv('bert_attempt/datasets/labels_wide.csv', index=False)

labels.sum(axis=0)
labels.sum(axis=0).plot.bar()
plt.savefig('example.pdf')
plt.clf()

########################

# strating to train

X = pd.read_csv('bert_attempt/datasets/text_column.csv')
y = pd.read_csv('bert_attempt/datasets/labels_wide.csv').values
labels_names = pd.read_csv('bert_attempt/datasets/labels_wide.csv').columns

labels_names
X
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

del X, y

text = X_test.copy()
mylabels = y_test.copy()

X_train = X_train['text'].tolist()
X_test = X_test['text'].tolist()

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open('bert_attempt/datasets/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(256)(embedding_layer)
dense_layer_1 = Dense(71, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC(), 'acc'])
print(model.summary())

history = model.fit(X_train, y_train, batch_size=128,
                    epochs=1, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[2])
model.save('my_model_1.h5')
model = load_model('my_model_1.h5')

text[50:100]
print(X_test[50:100])

predictions = model.predict(X_test[50:100], verbose=1)
predictio

ind = np.argpartition(predictions, -4, axis=1)[:,-4:]
ind

predictions[ind]

np.argmax(predictions, axis=1)

import matplotlib.pyplot as plt

history.history

plt.plot(history.history['auc_1'])
plt.plot(history.history['val_auc_1'])

plt.title('model accuracy')
plt.ylabel('auc_1')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.savefig('auc.pdf')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
