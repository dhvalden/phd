import pandas as pd
import text2emotion as te
import numpy as np
from sklearn.metrics import f1_score


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    for label in np.unique(labels):
        y_preds = preds[labels == label]
        y_true = labels[labels == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


df = pd.read_csv('./data/val_dataset.csv')

texts = df['text'].to_list()

classification = []

for ele in texts:
    out_dict = te.get_emotion(ele)
    out = max(out_dict, key=out_dict.get)
    classification.append(out)

classification = pd.Series(classification)

classification.unique()
label_dict_b = {'Sad': 0, 'Angry': 1, 'Love': 2,
                'Surprise': 3, 'Fear': 4, 'Happy': 5}
label_dict = {'sadness': 0, 'anger': 1, 'love': 2,
              'surprise': 3, 'fear': 4, 'joy': 5}

df['class'] = classification.map(label_dict_b)

accuracy_per_class(df['class'], df['label'])

print(f1_score(df['label'], df['class'], average='weighted'))
