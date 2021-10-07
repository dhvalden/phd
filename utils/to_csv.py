#!/usr/bin/env python

import ujson as json
import pandas as pd
import argparse


def get_records(file_path):
    records = []
    with open(file_path, 'r') as file:
        for line in file:
            record = {}
            raw = json.loads(line)
            record['date'] = raw['date']
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
