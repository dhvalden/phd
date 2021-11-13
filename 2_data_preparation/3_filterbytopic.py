#!/usr/bin/env python3

import os
import argparse
import ujson
from itertools import product
from multiprocessing import Pool

"""
This script receives simple_* files and outputs json files by social movements
This spliting is done based on list of keywords and hashtags provided in .txt
"""


def validate_file(file_name):
    """
    validate file name and path.
    """
    MSG_INVALID_PATH = "Error: Invalid file path/name. Path %s does not exist."
    if not valid_path(file_name):
        print(MSG_INVALID_PATH % (file_name))
        quit()
    return


def valid_path(path):
    # validate file path
    return os.path.exists(path)


def filtertopic(input_file, qfile):

    counter = 0
    validate_file(input_file)
    outfile = os.path.basename(qfile) + "_" + os.path.basename(input_file)

    with open(input_file, "r", encoding="utf-8") as f,\
    open(outfile, "w", encoding="utf-8") as outf,\
    open(qfile, "r", encoding="utf-8") as qf:
        keywords = [line.rstrip() for line in qf]
        keywords = set(keywords)
        for line in f:
            tweet = ujson.loads(line)
            text = tweet["full_text"]
            try:
                if tweet["retweeted_status"]:
                    if tweet["retweeted_status"]["extended_tweet"]:
                        rt = tweet["retweeted_status"]["extended_tweet"]["full_text"]
                    else:
                        rt = tweet["retweeted_status"]["text"]
                else:
                    rt = None
            except KeyError:
                rt = None
                pass

            try:
                if tweet["quoted_status"]:
                    if tweet["quoted_status"]["extended_tweet"]:
                        qt = tweet["quoted_status"]["extended_tweet"]["full_text"]
                    else:
                        qt = tweet["quoted_status"]["text"]
                else:
                    qt = None
            except KeyError:
                qt = None
                pass

            try:
                if any(kw in text for kw in keywords):
                    outf.write(ujson.dumps(tweet, ensure_ascii=False)+"\n")
                    counter += 1
                    continue
            except TypeError:
                pass
            try:
                if any(kw in rt for kw in keywords):
                    outf.write(ujson.dumps(tweet, ensure_ascii=False)+"\n")
                    counter += 1
                    continue
            except TypeError:
                pass
            try:
                if any(kw in qt for kw in keywords):
                    outf.write(ujson.dumps(tweet, ensure_ascii=False)+"\n")
                    counter += 1
                    continue
            except TypeError:
                pass

    print("Done! {} tweets filtered in {}.".format(counter, input_file))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--inputf", nargs='+', type=str,
                        help="Input file(s)")

    parser.add_argument("-q", "--qfile", nargs='+', type=str,
                        help="Query file")

    args = parser.parse_args()

    filenames = args.inputf
    qfiles = args.qfile

    if args.inputf is not None:
        with Pool(3) as p:
            p.starmap(filtertopic, product(filenames, qfiles))


if __name__ == '__main__':
    main()
