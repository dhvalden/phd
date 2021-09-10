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

def emojicount(args):

    input_file = args.inputf
    output_file = args.outputf
    spinner = spinning_cursor()
    counter = 0
    spincounter = 0

    emjlist = []
    
    validate_file(input_file)
    sys.stdout.write("Counting emojis on %s... " % input_file)
    
    with open(input_file, "r", encoding="utf-8") as f,\
    open(output_file, "w", encoding="utf-8") as outf:
        for line in f:
            tweet = json.loads(line)
            text = tweet["full_text"]
            thisdate = datetime.strptime(tweet["created_at"],
                                         "%a %b %d %H:%M:%S %z %Y")
            for c in text:
                out = {}
                if c in emoji.UNICODE_EMOJI:
                    out["emj"] = c
                    out["date"] = str(thisdate)
                    emjlist.append(out)
                    counter += 1
            spincounter +=1
            if spincounter % 1000 == 0:
                sys.stdout.write(next(spinner))
                sys.stdout.flush()
                sys.stdout.write("\b")
        outf.write(json.dumps(emjlist, ensure_ascii=False)+"\n")
    sys.stdout.write("Done!. %s emojis found.\n" % counter)

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--inputf", type = str,
                        help = "Input file")

    parser.add_argument("-o", "--outputf", type = str,
                        help = "Output file")
        
    args = parser.parse_args()
        
    if args.inputf != None and args.outputf != None:
        emojicount(args)
                
if __name__ == '__main__':
	main()
