import pandas as pd
import spacy
from spacy.matcher import Matcher
import re
import string
import argparse
from collections import Counter


def remove_emoji(string):
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
    return emoji_pattern.sub(r'', string)


def get_entities(string, nlp, matcher):
    doc = nlp(string)
    matches = matcher(doc)
    out = []
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        ele = f'{string_id}-{span.text}'
        out.append(ele)
    out = dict(Counter(out).most_common(4))
    return list(out.keys())


def preprocess(texts):
    output = re.sub(r"http\S+|www\S+|https\S+", '', texts, flags=re.MULTILINE)
    output = re.sub(r'\@\w+|\#\w+', '', output)
    output = re.sub(r"(http|@)\S+", '', output)
    output = re.sub(r"amp", "", output)
    output = re.sub(r"::", ": :", output)
    output = re.sub(r"’", "'", output)
    output = re.sub(r"\:.*\:", "", output)
    output = re.sub(r"[!@#$&\'\":_;,?'“”\-.…]", "", output)
    output = re.sub(r'\W*\b\w{1,3}\b', '', output)
    # Remove punctuations
    output = output.translate(str.maketrans('', '', string.punctuation))
    output = remove_emoji(output)
    output = re.sub(r'\d', '', output)
    output = re.sub(r' +', ' ', output)
    output = output.lower()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_path', type=str)
    parser.add_argument('-o', '--out_path', type=str)
    parser.add_argument('-m', '--spacy_model', type=str)
    args = parser.parse_args()
    nlp = spacy.load(args.spacy_model)
    matcher = Matcher(nlp.vocab)
    # Add match IDs
    org = [{"ENT_TYPE": "ORG"}]
    gpe = [{"ENT_TYPE": "GPE"}]
    fac = [{"ENT_TYPE": "FAC"}] # add keywords labels for specific movements
    loc = [{"ENT_TYPE": "LOC"}] # recover the PERSON
    matcher.add("ORG", [org])
    matcher.add("GPE", [gpe])
    matcher.add("FAC", [fac])
    matcher.add("LOC", [loc])

    df = pd.read_csv(args.file_path,
                     usecols=['full_text'],
                     lineterminator='\n')
    df['full_text'] = df['full_text'].astype(str)
    df['clean_text'] = df['full_text'].apply(preprocess)
    texts = df['clean_text'].to_list()

    ents = [get_entities(ele, nlp, matcher) for ele in texts]
    output = pd.DataFrame(ents, columns=['Entity-1', 'Entity-2',
                                         'Entity-3', 'Entity-4'])
    output.to_csv(args.out_path)


if __name__ == '__main__':
    main()
