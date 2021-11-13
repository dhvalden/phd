#!/usr/bin/env python3

import argparse
import pandas as pd


def filtertopic(args):
    input_file = args.inputf
    qfile = args.qfile
    out_file = args.outf
    df = pd.read_csv(input_file, index_col=0)
    with open(qfile, "r", encoding="utf-8") as qf:
        keywords = [line.rstrip() for line in qf]
        keywords = set(keywords)
        fltr = df['full_text'].str.contains('|'.join(keywords))
        fltr.fillna(False, inplace=True)
        df[fltr.values].to_csv(out_file)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--inputf", type=str,
                        help="Input file")

    parser.add_argument("-q", "--qfile", type=str,
                        help="Query file")

    parser.add_argument("-o", "--outf", type=str,
                        help="Output file")

    args = parser.parse_args()

    filtertopic(args)


if __name__ == '__main__':
    main()
