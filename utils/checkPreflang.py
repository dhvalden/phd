import ujson as json
import pandas as pd
import argparse

def checklang(args):

    langlist = []
    input_file = args.inputf

    print("Checking %s... " % input_file)

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            tweet = json.loads(line)
            langlist.append(tweet["lang"])

    count = pd.Series(langlist).value_counts(normalize=True)
    print(count)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--inputf", type=str,
                        help="Input file")

    args = parser.parse_args()

    if args.inputf is not None:
        checklang(args)


if __name__ == '__main__':
    main()
