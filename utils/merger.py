import pandas as pd

df1 = pd.read_csv('climate_plus_ner.csv',
                  lineterminator='\n',
                  usecols=['date', 'lang',
                           'groupbased-motivator',
                           'collective-action',
                           'ingroup', 'outgroup'])
df2 = pd.read_csv('labels_all_climate.csv',
                  lineterminator='\n',
                  index_col=0,
                  names=['class'])

pd.concat([df1, df2], axis=1).to_csv('climate_ner_emo.csv')
