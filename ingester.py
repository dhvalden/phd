#!/usr/bin/env python3
import os
import ujson as json
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


def ingester(line):
    tweet = json.loads(line)
    digest = {}
    if "id_str" in tweet:
        digest["id"] = tweet["id_str"]
    elif "id" in tweet:
        digest["id"] = str(tweet["id"])
    else:
        digest["id"] = None
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
    digest["retweeted_status"] = {}
    digest["quoted_status"] = {}

    if "retweeted_status" in tweet:
        digest["retweeted_status"]["text"] = tweet.get("retweeted_status")\
                                                  .get("text")
        try:
            digest["retweeted_status"]["full_text"] = tweet.get("retweeted_status")\
                                                           .get("extended_tweet")\
                                                           .get("full_text")
        except AttributeError:
            digest["retweeted_status"]["full_text"] = None
    else:
        digest["retweeted_status"] = None

    if "quoted_status" in tweet:
        digest["quoted_status"]["text"] = tweet.get("quoted_status")\
                                               .get("text")
        try:
            digest["quoted_status"]["full_text"] = tweet.get("quoted_status")\
                                                        .get("extended_tweet")\
                                                        .get("full_text")
        except AttributeError:
            digest["quoted_status"]["full_text"] = None
    else:
        digest["quoted_status"] = None

    digest["hashtags"] = tweet["entities"]["hashtags"]
    digest["urls"] = tweet["entities"]["urls"]
    digest["user_mentions"] = tweet["entities"]["user_mentions"]
    digest["text"] = tweet["text"]
    if "extended_tweet" in tweet:
        digest["extended_text"] = tweet.get("extended_tweet").get("full_text")
    else:
        digest["extended_text"] = None
    digest["source"] = tweet["source"]
    digest["lang"] = tweet["lang"]

    return(digest)


def writer(args):

    input_file = args.input_file
    output_file = args.output_file

    counter = 0
    
    validate_file(input_file)

    sys.stdout.write("Processing... ")

    with open(input_file, "r", encoding="utf-8") as f,\
         open(output_file, "w", encoding="utf-8") as out:
        for line in f:
            try:
                output = ingester(line)
            except KeyError:
                continue
            out.write(json.dumps(output, ensure_ascii=False)+"\n")
            counter += 1
    sys.stdout.write("Done!. %s objects processed.\n" % counter)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str,
                        help="Input file")

    parser.add_argument("output_file", type=str,
                        help="Output file")

    # parse the arguments from standard input
    args = parser.parse_args()

    if args.input_file is not None and args.output_file is not None:
        writer(args)


if __name__ == '__main__':
    main()
