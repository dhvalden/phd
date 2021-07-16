import pandas as pd

df = pd.read_csv('./data/merge_emo_data.csv', names=['text', 'text_label'])

df.head()

df['text_label'].()
