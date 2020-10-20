import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from time import time
sns.set_style('whitegrid')

def preprocess(texts, quiet=False):
    start = time()
    # Lowercasing
    texts = texts.str.lower()

    # Remove special chars
    texts = texts.str.replace(r"(http|@)\S+", "")
    #texts = texts.apply(demojize)
    texts = texts.str.replace(r"amp", "")
    texts = texts.str.replace(r"::", ": :")
    #texts = emoji.get_emoji_regexp().sub(u'', texts)
    texts = texts.str.replace(r"’", "'")
    #texts = texts.str.replace(r"[^a-z\':_]", " ")
    texts = texts.str.replace(r"\:.*\:", "")
    texts = texts.str.replace(r"[!@#$&\'\":_;,?'“”\-.…]", "")

    # Remove repetitions
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    texts = texts.str.replace(pattern, r"\1")

    # Transform short negation form
    texts = texts.str.replace(r"(can't|cannot)", 'can not')
    texts = texts.str.replace(r"n't", ' not')
    texts = texts.str.replace(r"rt", "")

    # Remove stop words
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.remove('not')
    stopwords.remove('nor')
    stopwords.remove('no')
    texts = texts.apply(lambda x: ' '.join(
        [word for word in str(x).split() if word not in stopwords]))

    if not quiet:
        print("Time to clean up: {:.2f} sec".format(time() - start))

    return texts


rawdata = pd.read_csv('bert_attempt/datasets/labeled.csv', dtype=str)
rawdata = rawdata.fillna('')
rawdata = rawdata.astype(str)
rawdata.head()

climatekeywords = []
brexitkeywords = []
hkkeywords = []
yvkeywords = []
usakeywords = []

with open('/home/daniel/climate_keywords.txt', 'r') as f:
    for line in f:
        climatekeywords.append(re.escape(line.strip()))

with open('/home/daniel/uk_keywords.txt', 'r') as f:
    for line in f:
        brexitkeywords.append(re.escape(line.strip()))

with open('/home/daniel/hk_keywords.txt', 'r') as f:
    for line in f:
        hkkeywords.append(re.escape(line.strip()))

with open('/home/daniel/yellowvest_keywords.txt', 'r') as f:
    for line in f:
        yvkeywords.append(re.escape(line.strip()))

with open('/home/daniel/usa_keywords.txt', 'r') as f:
    for line in f:
        usakeywords.append(re.escape(line.strip()))

climate = rawdata[rawdata['text'].str.contains('|'.join(climatekeywords),
                                               na=False)]

brexit = rawdata[rawdata['text'].str.contains('|'.join(brexitkeywords),
                                              na=False)]

hk = rawdata[rawdata['text'].str.contains('|'.join(hkkeywords),
                                          na=False)]

yv = rawdata[rawdata['text'].str.contains('|'.join(yvkeywords),
                                          na=False)]

usa = rawdata[rawdata['text'].str.contains('|'.join(usakeywords),
                                           na=False)]


climate['text_processed'] = preprocess(climate['text'])
yv['text_processed'] = preprocess(yv['text'])
hk['text_processed'] = preprocess(hk['text'])
usa['text_processed'] = preprocess(usa['text'])



def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))
    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.savefig('10wordsclimate.png')


# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(token_pattern=r'[^\s]+')
count_data = count_vectorizer.fit_transform(
    climate['text_processed'])

# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)


def get_lda_summary(number_topics, number_words, data_series, output_name):
    from sklearn.decomposition import LatentDirichletAllocation as LDA
    from pyLDAvis import sklearn as sklearn_lda
    import pickle
    import pyLDAvis
    count_vectorizer = CountVectorizer(token_pattern=r'[^\s]+')
    count_data = count_vectorizer.fit_transform(data_series)
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)
    if 1 == 1:
        LDAvis_prepared = sklearn_lda.prepare(lda,
                                              count_data, count_vectorizer)
    with open(output_name, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

        pyLDAvis.save_html(LDAvis_prepared,
                           output_name + '.html')


get_lda_summary(5, 10, climate['text_processed'], 'climate_lda')
get_lda_summary(5, 10, yv['text_processed'], 'yv_lda')
get_lda_summary(5, 10, hk['text_processed'], 'hk_lda')
get_lda_summary(5, 10, usa['text_processed'], 'usa_lda')

##########
## guided lda
##########

import guidedlda
vocab = count_vectorizer.get_feature_names()
word2id = dict((v, idx) for idx, v in enumerate(vocab))

seed_topic_list = [['😢'], ['😡', '😠'],
                   ['😀', '☺️'], ['🤣', '😂']]

model = guidedlda.GuidedLDA(n_topics=4, n_iter=100, random_state=7, refresh=20)

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(count_data, seed_topics=seed_topics, seed_confidence=0.15)

n_top_words = 10
topic_word = model.topic_word_

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words +
                                                             1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

# calculate doc lengths as the sum of each row of the dtm
doc_lengths = count_data.sum(axis=1)
doc_lengths = doc_lengths.flatten()
doc_lengths = doc_lengths.tolist()[0]
len(doc_lengths)
# transpose the dtm and get a sum of the overall term frequency
dtm_trans = count_data.T
total = dtm_trans.sum(axis=1)
total = total.flatten()
total = total.tolist()[0]
len(total)
len(vocab)

data = {
    'topic_term_dists': model.topic_word_,
    'doc_topic_dists': model.doc_topic_,
    'doc_lengths': doc_lengths,
    'vocab': vocab,
    'term_frequency': list(total)
}
# prepare the data
tef_vis_data = pyLDAvis.prepare(**data)

# this bit needs to be run after running the earlier code for reasons
pyLDAvis.display(tef_vis_data)

pyLDAvis.save_html(tef_vis_data, './guidedldavis_prepared_250k' + '.html')
