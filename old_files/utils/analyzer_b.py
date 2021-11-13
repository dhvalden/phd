import ujson
import argparse
import pandas as pd
from datetime import datetime
from multiprocessing import Pool


def analize(filepath):
    date = []
    topics = []
    with open(filepath, 'r') as f:
        for line in f:
            record = ujson.loads(line)
            thisdate = datetime.strptime(record['date'],
                                         '%Y-%m-%d %H:%M:%S+00:00')
            date.append(thisdate)
            topics.append(record['topics'])
    df1 = pd.DataFrame.from_records(topics)
    df1 = df1.astype(float)
    df2 = pd.DataFrame({'date': date})
    df = pd.concat([df2, df1], axis=1)
    df = df.sort_values(by='date')
    df = df.set_index('date')
    dfidmax = df.iloc[:, 0:].idxmax(axis=1)
    dfmax = df.max(axis=1)
    df = pd.concat([dfidmax, dfmax], axis=1)
    df.columns = ['topic', 'score']
    df['topic'] = df.topic.astype('category')
    df = df.loc[df['score'] > 0.39]
    df = df.reset_index()
    df = df[['date', 'topic']]
    df = df.groupby([df['date'].dt.round('H'), 'topic']).count()
    df = df.unstack(level=-1)
    oldcolnames = df.columns.values
    newcolnames = [item[1] for item in oldcolnames]
    df.columns = newcolnames
    dfnorm = df.div(df.sum(axis=1), axis=0)
    df.to_csv(str(filepath) + '_cat.csv')
    dfnorm.to_csv(str(filepath) + '_cat_norm.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='+', type=str)
    args = parser.parse_args()
    filenames = args.path
    if args.path is not None:
        with Pool(4) as p:
            p.map(analize, filenames)


if __name__ == '__main__':
    main()
