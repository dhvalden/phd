import pandas as pd

df1 = pd.read_csv('all_climate.csv',
                  lineterminator='\n',
                  usecols=['date',
                           'full_text',
                           'is_retweet',
                           'user'])
df2 = pd.read_csv('probs_labels_climate.csv',
                  lineterminator='\n',
                  index_col='id')

labels = {'sadness': '0', 'anger': '1', 'fear': '2', 'joy': '3'}
inverted_labels = {v: k for k, v in labels.items()}
df2.rename(columns=inverted_labels, inplace=True)

print(df1)
print(df2)

df3 = pd.concat([df1, df2], axis=1)

print(df3)

df3.to_csv('climate_emo.csv')
