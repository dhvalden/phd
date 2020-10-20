import re
import nltk
import emoji
from time import time
from emoji import demojize

def preprocess(texts, quiet=False):
  start = time()
  # Lowercasing
  texts = texts.str.lower()

  # Remove special chars
  texts = texts.str.replace(r"(http|@)\S+", "")
  texts = texts.apply(demojize)
  texts = texts.str.replace(r"::", ": :")
  #texts = emoji.get_emoji_regexp().sub(u'', texts)
  texts = texts.str.replace(r"’", "'")
  texts = texts.str.replace(r"[^a-z\':_]", " ")
  texts = texts.str.replace(r"\:.*\:", "")

  # Remove repetitions
  pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
  texts = texts.str.replace(pattern, r"\1")

  # Transform short negation form
  texts = texts.str.replace(r"(can't|cannot)", 'can not')
  texts = texts.str.replace(r"n't", ' not')
  texts = texts.str.replace(r"rt", "")

  # Remove stop words
  stopwords = nltk.corpus.stopwords.words('english')
  #stopwords.remove('not')
  #stopwords.remove('nor')
  #stopwords.remove('no')
  texts = texts.apply(
    lambda x: ' '.join([word for word in str(x).split() if word not in stopwords])
  )

  if not quiet:
    print("Time to clean up: {:.2f} sec".format(time() - start))

  return texts
