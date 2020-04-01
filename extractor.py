#!/usr/bin/env python3

import os
import sys
import argparse
import json
import emoji
from datetime import datetime

def validate_file(file_name):
    """
    validate file name and path.
    """
    MSG_INVALID_PATH = "Error: Invalid file path/name. Path %s does not exist."
    if not valid_path(file_name):
        print(MSG_INVALID_PATH%(file_name))
        quit()
    return


def valid_path(path):
    # validate file path
    return os.path.exists(path)

def spinning_cursor():
    while True:
        for cursor in "|/-\\":
            yield cursor

def emojize_dict_keys(myfile):
    with open(myfile, "r", encoding="utf-8") as f:
        query_dict = json.load(f)
        emojis = [emoji.emojize(w) for w in query_dict.keys()]
        return emojis
    

def emojiextract(args):

    input_file = args.inputf
    output_file = args.outputf
    query_file = args.queryf
    spinner = spinning_cursor()
    counter = 0
    spincounter = 0
    
    validate_file(input_file)
    validate_file(query_file)
    sys.stdout.write("Finding emojis on %s... " % input_file)
    
    with open(input_file, "r", encoding="utf-8") as f,\
    open(output_file, "w", encoding="utf-8") as outf:
        emojis = emojize_dict_keys(query_file)
        for line in f:
            tweet = json.loads(line)
            if tweet["lang"] == "en":
                text = tweet["full_text"]
                out = {}
                for c in text:
                    if c in emojis:
                        out["id"] = tweet["id"]
                        thisdate = datetime.strptime(tweet["created_at"],
                                                     "%a %b %d %H:%M:%S %z %Y")
                        out["date"] = str(thisdate)
                        out["user"] = tweet["screen_name"]
                        out["text"] = text
                        out["lang"] = tweet["lang"]
                        outf.write(json.dumps(out, ensure_ascii=False)+"\n")
                        counter += 1
                        break
            spincounter +=1
            if spincounter % 1000 == 0:
                sys.stdout.write(next(spinner))
                sys.stdout.flush()
                sys.stdout.write("\b")
    sys.stdout.write("Done!. %s tweets with emojis found.\n" % counter)

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--inputf", type = str,
                        help = "Input file")

    parser.add_argument("-o", "--outputf", type = str,
                        help = "Output file")

    parser.add_argument("-q", "--queryf", type = str,
                        help = "Query file")
        
    args = parser.parse_args()
        
    if args.inputf != None and args.outputf != None:
        emojiextract(args)
                
if __name__ == '__main__':
	main()
