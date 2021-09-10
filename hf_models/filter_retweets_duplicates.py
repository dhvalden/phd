import argparse
import hashlib
import ujson as json


def myfilter(filepath):
    hashes = set()
    with open(filepath, 'r') as f:
        for line in f:
            tweet: str = json.loads(line)
            text: str = tweet["full_text"]
            hashed_text = hashlib.sha256(text.encode('utf-8'))
            if hashed_text in hashes:
                continue
            else:
                hashes.add(hashed_text)
            if text.startswith("RT "):
                continue
            else:
                print(json.dumps(tweet))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    filename = args.path
    if args.path is not None:
        myfilter(filename)


if __name__ == '__main__':
    main()
