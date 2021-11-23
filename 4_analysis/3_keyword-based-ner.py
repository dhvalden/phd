import argparse
import re
import string
import json
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter


mystopwords = stopwords.words('english')
tweet_tokenizer = TweetTokenizer()


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


def preprocess(tweet, stopwords=mystopwords):
    tweet = remove_emoji(tweet)
    tweet = tweet.lower()
    # Remove urls
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\brt\b', '', tweet)
    tweet = re.sub(r"’", "", tweet)
    tweet = re.sub(r"amp", "", tweet)
    tweet = re.sub(r"…", "", tweet)
    tweet = re.sub(r'“', '', tweet)
    tweet = re.sub(r'”', '', tweet)
    tweet = re.sub(r'\W*\b\w{1,3}\b', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = tweet_tokenizer.tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if word not in stopwords]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
    return " ".join(lemma_words)


def get_value(mystring, mydict):
    output = []
    for key in mydict:
        if key in mystring:
            output.append(mydict[key])
    output = dict(Counter(output))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str)
    parser.add_argument('-k', '--keywords_file', type=str)
    parser.add_argument('-m', '--s_movement', type=str)
    parser.add_argument('-o', '--output_file', type=str)

    args = parser.parse_args()
    df = pd.read_csv(args.input_file, lineterminator='\n')
    df['full_text'] = df['full_text'].astype(str)
    df['clean_text'] = df['full_text'].apply(preprocess)
    clean_tweets = df['clean_text'].to_list()

    with open(args.keywords_file) as kfile:
        data = json.load(kfile)
        keywords = data['movements'][args.s_movement]
        out = [get_value(tweet, keywords) for tweet in clean_tweets]

    result_df = pd.DataFrame.from_records(out).fillna(0).astype(int)

    pd.concat([df, result_df], axis=1).to_csv(args.output_file)


if __name__ == '__main__':
    main()
