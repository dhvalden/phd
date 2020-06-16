import pandas as pd
import pickle
from nlp import Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer

dataset_path = '../datasets/labeled.csv'
dataset = Dataset(dataset_path)
dataset.load()
dataset.preprocess_texts()

num_words = 10000

tokenizer = Tokenizer(num_words=num_words, lower=True)
tokenizer.fit_on_texts(dataset.cleaned_data.text)

file_to_save = '../datasets/tokenizer.pickle'
with file_to_save.open('wb') as file:
    pickle.dump(tokenizer, file)

count = dataset.cleaned_data['text'].str.split().str.len()
data = dataset.cleaned_data[count > 1]
ia = data[data['label'] == 'This is not norma…'].index
ib = data[data['label'] == '!'].index
data = data.drop(ia)
data = data.drop(ib)

train = pd.DataFrame(columns=['label', 'text'])
validation = pd.DataFrame(columns=['label', 'text'])
for label in data.label.unique():
    label_data = data[data.label == label]
    train_data, validation_data = train_test_split(label_data, test_size=0.30,
                                                   train_size=0.7)
    train = pd.concat([train, train_data])
    validation = pd.concat([validation, validation_data])

input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
num_classes = len(data.label.unique())
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

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

train_sequences = [text.split() for text in train.text]
validation_sequences = [text.split() for text in validation.text]
list_tokenized_train = tokenizer.texts_to_sequences(train_sequences)
list_tokenized_validation = tokenizer.texts_to_sequences(validation_sequences)
x_train = pad_sequences(list_tokenized_train, maxlen=input_length)
x_validation = pad_sequences(list_tokenized_validation, maxlen=input_length)

encoder = LabelBinarizer()
encoder.fit(data.label.unique())

encoder_path = '../emotion_recognition/encoder.pickle'
with encoder_path.open('wb') as file:
    pickle.dump(encoder, file)

y_train = encoder.transform(train.label)
y_validation = encoder.transform(validation.label)

batch_size = 128
epochs = 1

model.fit(
    x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_validation, y_validation)
)

model.save_weights('../model_weights_general.h5')
