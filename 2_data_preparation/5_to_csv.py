#!/usr/bin/env python

import ujson as json
import pandas as pd
import argparse

"""
This scripts converts social movements json dats files to csv format
"""


def get_records(file_path):
    records = []
    with open(file_path, 'r') as file:
        for line in file:
            record = {}
            raw = json.loads(line)
            record['date'] = raw['date']
            if raw['retweeted_status'] is not None:
                record['is_retweet'] = True
                if 'full_text' in raw['retweeted_status']:
                    if raw['retweeted_status']['full_text'] is not None:
                        record['full_text'] = raw['retweeted_status']['full_text']
                else:
                    record['full_text'] = raw['retweeted_status']['text']
            else:
                record['is_retweet'] = False
                record['full_text'] = raw['full_text']
            record['user'] = raw['user']
            record['lang'] = raw['lang']
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()
    myrecords = get_records(args.file_path)
    df = pd.DataFrame.from_records(myrecords)
    df = df.dropna()
    df['full_text'] = df['full_text'].replace(r'\n', ' ', regex=True)
    df.to_csv(args.out_path, index=False)


if __name__ == '__main__':
    main()
