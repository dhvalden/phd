#!/usr/bin/env python3

import os
import sys
import argparse
import ujson
from datetime import datetime
from multiprocessing import Pool


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


def simplify(input_file):

    counter = 0

    validate_file(input_file)
    sys.stdout.write("Simplifying %s... " % input_file)

    with open(input_file, "r", encoding="utf-8") as f,\
    open("simple_" + input_file, "w", encoding="utf-8") as outf:
        for line in f:
            tweet = ujson.loads(line)
            out = {}
            thisdate = datetime.strptime(tweet["created_at"],
                                         "%a %b %d %H:%M:%S %z %Y")
            out["date"] = str(thisdate)
            out["user"] = tweet["screen_name"]
            if tweet["extended_text"] is not None:
                out["full_text"] = tweet["extended_text"]
            else:
                out["full_text"] = tweet["text"]
            out["lang"] = tweet["lang"]
            out["hashtags"] = tweet["hashtags"]
            out["user_mentions"] = tweet["user_mentions"]
            out["retweeted_status"] = tweet["retweeted_status"]
            out["quoted_status"] = tweet["quoted_status"]
            outf.write(ujson.dumps(out, ensure_ascii=False)+"\n")
            counter += 1

    sys.stdout.write("Done!. %s tweets simplified.\n" % counter)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--inputf", nargs='+', type=str,
                        help="Input file(s)")

    args = parser.parse_args()

    filenames = args.inputf

    if args.inputf is not None:
        with Pool(3) as p:
            p.map(simplify, filenames)


if __name__ == '__main__':
    main()
