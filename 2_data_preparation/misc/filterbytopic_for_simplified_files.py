#!/usr/bin/env python3

import argparse
import ujson


def filtertopic(args):
    input_file = args.inputf
    qfile = args.qfile

    with open(input_file, "r", encoding="utf-8") as f,\
         open(qfile, "r", encoding="utf-8") as qf:
        keywords = [line.rstrip() for line in qf]
        keywords = set(keywords)
        for line in f:
            tweet = ujson.loads(line)
            text = tweet["full_text"]
            try:
                if any(kw in text for kw in keywords):
                    print(ujson.dumps(tweet))
            except TypeError:
                pass


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--inputf", type=str,
                        help="Input file")

    parser.add_argument("-q", "--qfile", type=str,
                        help="Query file")

    args = parser.parse_args()

    if args.inputf is not None:
        filtertopic(args)


if __name__ == '__main__':
    main()
