import warnings
import gensim
import ujson as json
import helpers as hp
import argparse
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from gensim.corpora import Dictionary

warnings.filterwarnings(action='ignore', category=UserWarning)

mystopwords = stopwords.words('english')
mystopwords.remove('not')
mystopwords.remove('nor')
mystopwords.remove('no')
newStopWords = ['。', '，', '、', '！', 'hong', 'kong', 'police', '🇭', '🇰', '：', None]
mystopwords.extend(newStopWords)

tweet_tokenizer = TweetTokenizer()


def labeler(args):

    input_path = args.input_path
    model_path = args.model_path
    output_path = args.output_path
    dictfile_path = args.dictfile_path

    # Load model
    lda = gensim.models.LdaModel.load(model_path)
    samplecorp = []
    with open(dictfile_path, 'r') as df:
        for line in df:
            tweet = json.loads(line)
            text = hp.preprocess_tweet_text(tweet['full_text'], stopw=mystopwords)
            samplecorp.append(text.split())
    dictionary = Dictionary(samplecorp)

    with open(input_path, 'r') as f,\
         open(output_path, "w", encoding="utf-8") as o:
        for line in f:
            output = {}
            tweet = json.loads(line)
            text = hp.preprocess_tweet_text(tweet['full_text'], stopw=mystopwords)
            corp = text.split()
            bow = dictionary.doc2bow(corp)
            try:
                topics = dict(lda[bow])
            except:
                continue
            topics = {str(key): str(value) for key, value in topics.items()}
            output['topics'] = topics
            output['date'] = tweet['date']
            output['words'] = corp
            output['full_text'] = tweet['full_text']
            o.write(json.dumps(output, ensure_ascii=False)+"\n")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_path", type=str,
                        help="Input file")
    parser.add_argument("-o", "--output_path", type=str,
                        help="Output file")
    parser.add_argument("-d", "--dictfile_path", type=str,
                        help="Sample file to create dictionary")
    parser.add_argument("-m", "--model_path", type=str,
                        help="Model file")
    args = parser.parse_args()

    labeler(args)


if __name__ == '__main__':
    main()
