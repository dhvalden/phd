import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('~/code/phd/data/climate_ner_emo.csv', index_col=0)
df = df.set_index(pd.to_datetime(df.date.values))

df = df.drop(columns=['date', 'lang'])

labels = {'sadness': 0, 'anger': 1, 'fear': 2, 'joy': 3}
invert_labels = {v: k for k, v in labels.items()}
df['class'] = df['class'].replace(invert_labels)
df = pd.concat([df, pd.get_dummies(df['class'])],
               axis=1).drop(columns=['class'])


events = pd.read_csv('~/code/phd/data/xr_events.csv',
                     usecols=['event_date', 'event_type', 'sub_event_type'])
events = events.set_index(pd.to_datetime(events.event_date.values, utc=True))


df_gbm = df.loc[(df['groupbased-motivator'] > 1)]
df_ca = df.loc[(df['collective-action'] > 1)]
df_in = df.loc[(df['ingroup'] > 1)]
df_out = df.loc[(df['outgroup'] > 0)]


def get_plot(emo, events):
    df1 = emo.iloc[:, 4:8].resample("D").sum()
    df2 = events['event_type'].resample('D').count()
    merge = pd.merge(df1, df2, how='inner', left_index=True, right_index=True)
    x1 = merge.iloc[:, :4].resample("3D").sum()
    x2 = merge['event_type'].resample("3D").sum()
    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1)
    sns.lineplot(data=x1, ax=ax1)
    sns.lineplot(data=x2, ax=ax2)
    plt.show()


get_plot(df_gbm, events)

get_plot(df_ca, events)

get_plot(df_in, events)

get_plot(df_out, events)
