import argparse
import re
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002500-\U00002BEF"
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def preprocess(tweet):
    tweet = remove_emoji(tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\brt\b', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r"’", "", tweet)
    tweet = re.sub(r"amp", "", tweet)
    tweet = re.sub(r"…", "", tweet)
    tweet = re.sub(r":", "", tweet)
    tweet = re.sub(r'“', '', tweet)
    tweet = re.sub(r'”', '', tweet)
    return tweet


def get_nouns(tweet, pattern, tagger):
    nouns = []
    clean_text = preprocess(tweet)
    sentence = Sentence(clean_text)
    tagger.predict(sentence)
    sent_dict = sentence.to_dict(tag_type='pos')
    for ent in sent_dict['entities']:
        if re.match(pattern, ent['labels'][0].value):
            nouns.append(ent['text'])
    return ' '.join(nouns)


def main():
    tagger = SequenceTagger.load('pos-fast')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_path', type=str)
    parser.add_argument('-o', '--out_path', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.file_path,
                     usecols=['full_text'],
                     lineterminator='\n')
    tweets = df['full_text'].astype(str).to_list()
    pattern = re.compile('NN|NN.')
    out_nouns = [get_nouns(tweet, pattern, tagger) for tweet in tweets]
    result_df = pd.DataFrame({'index': df.index, 'nouns': out_nouns})
    result_df.to_csv(args.out_path, index=False)


if __name__ == '__main__':
    main()
