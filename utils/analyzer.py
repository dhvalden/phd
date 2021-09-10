import ujson
import argparse
import pandas as pd
from datetime import datetime
from multiprocessing import Pool


def analize(filepath):
    date = []
    sentiments = []
    with open(filepath, 'r') as f:
        for line in f:
            record = ujson.loads(line)
            thisdate = datetime.strptime(record['date'],
                                         '%Y-%m-%d %H:%M:%S+00:00')
            sents = record['sentiments']
            sents = sents.replace("\'", "\"")
            sents = ujson.loads(sents)
            date.append(thisdate)
            sentiments.append(sents)
    df1 = pd.DataFrame.from_records(sentiments)
    df1 = df1.astype(float)
    df2 = pd.DataFrame({'date': date})
    df = pd.concat([df2, df1], axis=1)
    df = df.sort_values(by='date')
    df = df.set_index('date')
    df = df.resample('H')
    dfm = df.agg('mean').add_suffix('_mean')
    dfc = df.agg('count').add_suffix('_count')
    df = pd.concat([dfm, dfc], axis=1)
    df.to_csv(str(filepath) + '.csv')


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
