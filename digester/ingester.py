#!/usr/bin/env python3
import os
import json
import sys
import argparse

# Utility funtions


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


def spinning_cursor():
    while True:
        for cursor in "|/-\\":
            yield cursor


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)


def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    nlines = sum(buf.count(b'\n') for buf in f_gen)
    sys.stdout.write("%s objects found. " % nlines)


def extract_text(tweet):
    full_text = "".encode()
    if "retweeted_status" in tweet:
        if "extended_tweet" in tweet["retweeted_status"]:
            full_text = tweet["retweeted_status"]["extended_tweet"]["full_text"]
        else:
            full_text = tweet["retweeted_status"]["text"]
    elif "quoted_status" in tweet:
        if "extended_tweet" in tweet["quoted_status"]:
            full_text = tweet["quoted_status"]["extended_tweet"]["full_text"]
        else:
            full_text = tweet["quoted_status"]["text"]
    elif "extended_tweet" in tweet:
        full_text = tweet["extended_tweet"]["full_text"]
    else:
        full_text = tweet["text"]
    return(full_text)


def ingester(line):
    tweet = json.loads(line)
    digest = {}
    digest["id"] = tweet["id_str"]
    digest["created_at"] = tweet["created_at"]
    digest["screen_name"] = tweet["user"]["screen_name"]
    digest["location"] = tweet["user"]["location"]
    digest["description"] = tweet["user"]["description"]
    digest["user_id"] = tweet["user"]["id_str"]
    digest["verified"] = tweet["user"]["verified"]
    digest["followers_count"] = tweet["user"]["followers_count"]
    digest["friends_count"] = tweet["user"]["friends_count"]
    digest["listed_count"] = tweet["user"]["listed_count"]
    digest["favourites_count"] = tweet["user"]["favourites_count"]
    digest["statuses_count"] = tweet["user"]["statuses_count"]
    digest["user_created_at"] = tweet["user"]["created_at"]
    digest["quote_count"] = tweet["quote_count"]
    digest["reply_count"] = tweet["reply_count"]
    digest["retweet_count"] = tweet["retweet_count"]
    digest["favorite_count"] = tweet["favorite_count"]
    digest["hashtags"] = tweet["entities"]["hashtags"]
    digest["urls"] = tweet["entities"]["urls"]
    digest["user_mentions"] = tweet["entities"]["user_mentions"]
    digest["full_text"] = extract_text(tweet)
    digest["text"] = tweet["text"]
    digest["extended_text"] = tweet.get("extended_tweet", {}).get("full_text")
    digest["source"] = tweet["source"]
    digest["lang"] = tweet["lang"]

    return(digest)


def writer(args):

    input_file = args.input_file
    output_file = args.output_file

    validate_file(input_file)

    counter = 0
    spinner = spinning_cursor()

    sys.stdout.write("Processing... ")

    with open(input_file, "r", encoding="utf-8") as f,\
         open(output_file, "w", encoding="utf-8") as out:
        for line in f:
            try:
                output = ingester(line)
                out.write(json.dumps(output, ensure_ascii=False)+"\n")
            except:
                continue
            counter += 1
            if counter % 1000 == 0:
                sys.stdout.write(next(spinner))
                sys.stdout.flush()
                sys.stdout.write("\b")

    sys.stdout.write("Done!. %s objects processed.\n" % counter)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str,
                        help="Input file")

    parser.add_argument("output_file", type=str,
                        help="Output file")

    parser.add_argument("-c", "--count", help="Count the number of objects",
                        action="store_true")

    # parse the arguments from standard input
    args = parser.parse_args()

    # calling functions depending on type of argument
    if args.count:
        rawgencount(args.input_file)

    if args.input_file is not None and args.output_file is not None:
        writer(args)


if __name__ == '__main__':
    main()
