import ujson as json
import argparse
import helpers as hp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

mystopwords = stopwords.words('english')
mystopwords.remove('not')
mystopwords.remove('nor')
mystopwords.remove('no')
newStopWords = []
mystopwords.extend(newStopWords)


def labeler(args):
    input_path = args.input_path
    analyzer = SentimentIntensityAnalyzer()
    with open(input_path, 'r') as f:
        for line in f:
            output = {}
            tweet = json.loads(line)
            text = hp.preprocess_tweet_text(tweet['full_text'],
                                            stopw=mystopwords)
            vs = analyzer.polarity_scores(text)
            output['sentiments'] = str(vs)
            output['date'] = tweet['date']
            output['full_text'] = tweet['full_text']
            print(json.dumps(output, ensure_ascii=False)+"\n")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_path", type=str,
                        help="Input file")
    args = parser.parse_args()

    labeler(args)

if __name__ == '__main__':
    main()
