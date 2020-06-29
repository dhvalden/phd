import pandas as pd
from sklearn.utils import class_weight
import numpy as np
from emoji import demojize
from tensorflow import keras
import bert
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import seaborn as sns
import os


rawdata = pd.read_csv('bert_attempt/datasets/preprocesed_data.csv',
                      index_col=0)
print(rawdata.head())

labels = rawdata['label'].apply(demojize)
labels = pd.Categorical(labels)
y = labels.codes
X = rawdata['text'].tolist()


plot = sns.countplot(labels).get_figure()
plot.savefig('plotcount.png')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=42)

# calculation class weights for heavely unbalanced sample.
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_train),
                                                  y_train)
class_weights = {i : class_weights[i] for i in range(71)}

class_weights
# y sets to one_hot format.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# loading bert pre-trained model and parameters
model_dir = 'bert_attempt/models/uncased_L-4_H-256_A-4'
model_name = "uncased_L-4_H-256_A-4"
model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
bert_params = bert.params_from_pretrained_ckpt(model_dir)
l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

# creating bert tokenizer
do_lower_case = not (model_name.find("cased") == 0 or model_name.find("multi_cased") == 0)
bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, model_ckpt)
vocab_file = os.path.join(model_dir, "vocab.txt")
tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
 
# helper function to apply to sequeces of text

def tokenize_reviews(texts):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(texts))

X_train = [tokenize_reviews(text) for text in X_train]
X_test = [tokenize_reviews(text) for text in X_test]

vocab_size = len(tokenizer.vocab)

maxlen = 280
numclass = 71

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# creating model: String with simple model to test result
# Pending including bert embeddings (if they exists) to improve models
# so far only using BERT tokenizer.

l_input_ids = keras.layers.Input(shape=(maxlen,), dtype='int32')
l_token_type_ids = keras.layers.Input(shape=(maxlen,), dtype='int32')

output = l_bert(l_input_ids)

LSTM_Layer_1 = keras.layers.LSTM(128)(output)
logits = keras.layers.Dense(numclass, activation='softmax')(LSTM_Layer_1)
model = keras.Model(inputs=l_input_ids, outputs=logits)
model.build(input_shape=(None, maxlen))

bert.load_bert_weights(l_bert, model_ckpt)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

history = model.fit(X_train, y_train, batch_size=128,
                    epochs=1, verbose=1, validation_split=0.2,
                    class_weight=class_weights)
