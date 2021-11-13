import pandas as pd
import pylab
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('~/code/phd/data/climate_emo_ner_xr_final.csv', index_col=0)
df = df.set_index(pd.to_datetime(df.date.values))
df = df.drop(columns=['Unnamed: 0.1', 'date', 'is_retweet',
                      'full_text', 'user', 'clean_text'])

fltr = df.iloc[:, :4] >= 0.8  # Section with emotion confidence values

df2 = df[fltr.any(1)]
df_b = df2.iloc[:, :4].idxmax(axis="columns")
df_c = df2.iloc[:, 4:]

df_final = pd.concat([df_c, pd.get_dummies(df_b)],
                     axis=1)

events = pd.read_csv('~/code/phd/data/xr_events_reconciliated.csv',
                     usecols=['event_date', 'scale_factor'])
events = events.set_index(pd.to_datetime(events.event_date.values, utc=True))
events = events.drop('event_date', 1)


df_gbm = df_final.loc[(df_final['groupbased-motivator'] > 1)]
df_ca = df_final.loc[(df_final['collective-action'] > 1)]
df_in = df_final.loc[(df_final['ingroup'] > 1)]
df_out = df_final.loc[(df_final['outgroup'] > 0)]


def get_plot(emo, events):
    df1 = emo.iloc[:, 4:].resample("D").sum()
    df2 = events.resample('D').sum()
    merge = pd.merge(df1, df2, how='inner', left_index=True, right_index=True)
    x1 = merge.iloc[:, :4].resample("3D").sum()
    x2 = merge['scale_factor'].resample("3D").sum()
    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1)
    sns.lineplot(data=x1, ax=ax1)
    sns.lineplot(data=x2, ax=ax2)
    plt.show()
    return x1, x2


get_plot(df_gbm, events)

get_plot(df_ca, events)

get_plot(df_in, events)

rollemo, rollevent = get_plot(df_out, events)

np.corrcoef(rollevent, rollemo['anger'])


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals=np.asarray(signals),
                avgFilter=np.asarray(avgFilter),
                stdFilter=np.asarray(stdFilter))


def get_anomalies(y, lag=5, threshold=3, influence=0.2):
    result = thresholding_algo(np.array(y),
                               lag=lag,
                               threshold=threshold,
                               influence=influence)
    pylab.subplot(211)
    pylab.plot(np.arange(1, len(y)+1), y)

    pylab.plot(np.arange(1, len(y)+1),
               result["avgFilter"], color="cyan", lw=2)

    pylab.plot(np.arange(1, len(y)+1),
               result["avgFilter"] + threshold * result["stdFilter"],
               color="green", lw=2)

    pylab.plot(np.arange(1, len(y)+1),
               result["avgFilter"] - threshold * result["stdFilter"],
               color="green", lw=2)

    pylab.subplot(212)
    pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
    pylab.ylim(-1.5, 1.5)
    pylab.show()
    return result['signals']


rollemo, rollevent = get_plot(df_out, events)

get_anomalies(rollemo['anger'], influence=0.1)
