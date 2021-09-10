import warnings
import gensim
import json
import re
import string
import numpy as np
import pandas as pd
import nltk
from gensim.models import LdaMulticore
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings(action='ignore', category=UserWarning)

mystopwords = stopwords.words('english')
mystopwords.remove('not')
mystopwords.remove('nor')
mystopwords.remove('no')
newStopWords = ['gop', 'na', 'gon', 'climate', 'people', 'today']
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


recordslist = []

with open('/home/daniel/code/phd/datasets/climatesample100k.json', 'r') as f:
    for line in f:
        recordslist.append(json.loads(line))

rawdata = pd.DataFrame.from_records(recordslist)
rawdata = rawdata.fillna('')
rawdata = rawdata.astype(str)

rawdata['text_processed'] = rawdata['full_text'].apply(preprocess_tweet_text)

corp = [d.split() for d in rawdata['text_processed']]

dictionary = gensim.corpora.Dictionary(corp)


def test_eta(eta, dictionary, ntopics, print_topics=True):
    np.random.seed(42) # set the random seed for repeatability
    bow = [dictionary.doc2bow(line) for line in corp] # get the bow-format lines with the set dictionary
    with (np.errstate(divide='ignore')):  # ignore divide-by-zero warnings
        model = LdaMulticore(
            corpus=bow, id2word=dictionary, num_topics=ntopics,
            random_state=42, chunksize=100, eta=eta,
            workers=3, eval_every=None, passes=1)
    if print_topics:
        # display the top terms for each topic
        for topic in range(ntopics):
            print('Topic {}: {}'.format(topic, [dictionary[w] for w, p in model.get_topic_terms(topic, topn=10)]))
    return model


def create_eta(priors, etadict, ntopics):
    eta = np.full(shape=(ntopics, len(etadict)), fill_value=1) # create a (ntopics, nterms) matrix and fill with 1
    for word, topic in priors.items(): # for each word in the list of priors
        keyindex = [index for index,term in etadict.items() if term==word] # look up the word in the dictionary
        if (len(keyindex)>0): # if it's in the dictionary
            eta[topic,keyindex[0]] = 1e7  # put a large number in there
            eta = np.divide(eta, eta.sum(axis=0)) # normalize so that the probabilities sum to 1 over all topics
    return eta


apriori_original = {
    '✊': 0, '💪': 0,  # Gambatte
    '🔥': 1, '💥': 1,  # Strike
    '🌎': 2, '💚': 2,  # Unity
    '😭': 3, '😔': 3,  # Help!
}


eta = create_eta(apriori_original, dictionary, 4)
model = test_eta(eta, dictionary, 4)
model.save('./climateModel')
