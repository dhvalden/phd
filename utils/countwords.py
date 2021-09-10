#!/usr/bin/env python3

import ujson as json
import argparse
import collections
import itertools
import helpers as hp
import pandas as pd
import matplotlib
matplotlib.use('agg')
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk import bigrams

import warnings
warnings.filterwarnings("ignore")

mystopwords = stopwords.words('english') + stopwords.words('french') + stopwords.words('spanish')
mystopwords.remove('not')
mystopwords.remove('nor')
mystopwords.remove('no')
newStopWords = []
mystopwords.extend(newStopWords)


def wordcount(args):

    infile = args.infile
    wordslist = []

    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            tweet = json.loads(line)
            words = hp.preprocess_unjointed(tweet['full_text'],
                                            stopw=mystopwords)
            wordslist.append(words)
    listofwords = [item for sublist in wordslist for item in sublist]
    freqwords = nltk.FreqDist(listofwords)
    print('Word count:')
    print(freqwords.most_common(50))
    fig = plt.figure(figsize=(20, 8))
    freqwords.plot(50, cumulative=False)
    fig.savefig(str(infile) + '_freqDist.png', bbox_inches="tight")


def get_bigrams(args):
    infile = args.infile
    wordslist = []

    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            tweet = json.loads(line)
            words = hp.preprocess_unjointed(tweet['full_text'],
                                            stopw=mystopwords)
            wordslist.append(words)
    terms_bigram = [list(bigrams(tweet)) for tweet in wordslist]
    mybigrams = list(itertools.chain(*terms_bigram))
    bigram_counts = collections.Counter(mybigrams)
    print('Bigrams:')
    print(bigram_counts.most_common(50))

    bigram_df = pd.DataFrame(bigram_counts.most_common(50),
                             columns=['bigram', 'count'])
    d = bigram_df.set_index('bigram').T.to_dict('records')
    G = nx.Graph()
    for k, v in d[0].items():
        G.add_edge(k[0], k[1], weight=(v * 10))
    fig, ax = plt.subplots(figsize=(20, 16))
    pos = nx.spring_layout(G, k=2)
    nx.draw_networkx(G, pos,
                     font_size=10,
                     width=3,
                     edge_color='grey',
                     node_color='purple',
                     with_labels=False,
                     ax=ax)
    for key, value in pos.items():
        x, y = value[0]+.02, value[1]+.02
        ax.text(x, y,
                s=key,
                bbox=dict(facecolor='red', alpha=0.25),
                horizontalalignment='center', fontsize=16)
    plt.savefig(str(infile) + '_bigramNet.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Input file')
    parser.add_argument('-wc', '--wordcount', help='Word count of tweets',
                        action="store_true")
    parser.add_argument('-bg', '--bigrams', help='Bigrams of tweets',
                        action="store_true")
    args = parser.parse_args()
    if args.infile is None:
        quit('No input file found')
    if args.wordcount:
        wordcount(args)
    if args.bigrams:
        get_bigrams(args)


if __name__ == '__main__':
    main()
