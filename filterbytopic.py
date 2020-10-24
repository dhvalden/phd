#!/usr/bin/env python3

import os
import sys
import argparse
import ujson


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


def filtertopic(args):

    input_file = args.inputf
    output_file = args.outputf
    qfile = args.qfile
    counter = 0

    validate_file(input_file)
    sys.stdout.write("Filtering %s... " % input_file)

    with open(input_file, "r", encoding="utf-8") as f,\
    open(output_file, "w", encoding="utf-8") as outf,\
    open(qfile, "r", encoding="utf-8") as qf:
        keywords = [line.rstrip() for line in qf]
        for line in f:
            tweet = ujson.loads(line)
            text = tweet["full_text"]
            if any(kw in text for kw in keywords):
                outf.write(ujson.dumps(tweet, ensure_ascii=False)+"\n")
                counter += 1

    sys.stdout.write("Done!. %s tweets filtered.\n" % counter)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--inputf", type=str,
                        help="Input file")
    parser.add_argument("-o", "--outputf", type=str,
                        help="Output file")
    parser.add_argument("-q", "--qfile", type=str,
                        help="Query file")

    args = parser.parse_args()

    if args.inputf is not None and args.outputf is not None:
        filtertopic(args)


if __name__ == '__main__':
    main()
