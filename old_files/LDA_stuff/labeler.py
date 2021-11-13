import warnings
import gensim
import ujson as json
import argparse
import re
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary

warnings.filterwarnings(action='ignore', category=UserWarning)

mystopwords = stopwords.words('english')
mystopwords.remove('not')
mystopwords.remove('nor')
mystopwords.remove('no')
newStopWords = ['gop', 'na', 'gon', 'climate', 'brexit']
mystopwords.extend(newStopWords)

tweet_tokenizer = TweetTokenizer()


def preprocess_tweet_text(tweet, stopwords=mystopwords):
    tweet = tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#\w+', '', tweet)
    tweet = re.sub(r"(http|@)\S+", "", tweet)
    tweet = re.sub(r"amp", "", tweet)
    tweet = re.sub(r"::", ": :", tweet)
    tweet = re.sub(r"’", "'", tweet)
    tweet = re.sub(r"\:.*\:", "", tweet)
    tweet = re.sub(r"[!@#$&\'\":_;,?'“”\-.…]", "", tweet)
    tweet = re.sub(r"(can't|cannot)", 'can not', tweet)
    tweet = re.sub(r"n't", ' not', tweet)
    tweet = re.sub(r"rt", "", tweet)
    tweet = re.sub(r'\W*\b\w{1,3}\b', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = tweet_tokenizer.tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if word not in stopwords]
    # ps = PorterStemmer()
    # stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
    return " ".join(lemma_words)


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
            text = preprocess_tweet_text(tweet['full_text'])
            samplecorp.append(text.split())
    dictionary = Dictionary(samplecorp)

    with open(input_path, 'r') as f,\
         open(output_path, "w", encoding="utf-8") as o:
        for line in f:
            output = {}
            tweet = json.loads(line)
            text = preprocess_tweet_text(tweet['full_text'])
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
