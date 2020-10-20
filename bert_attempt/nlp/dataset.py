import re
import pandas as pd
from time import time
from pathlib import Path
from .utils import preprocess

class Dataset:
  def __init__(self, filename, label_col='label', text_col='text'):
    self.filename = filename
    self.label_col = label_col
    self.text_col = text_col


#this property will not work when the input labels are not list
  @property
  def data(self):
    data = self.dataframe[self.label_col + self.text_col].copy()
    data.columns = self.label_col + self.text_col
    return data

  @property
  def cleaned_data(self):
    data = self.dataframe[self.label_col + ['cleaned']]
    data.columns = self.label_col + ['text']
    return data

  def load(self):
    df = pd.read_csv(Path(self.filename).resolve(), dtype=str)
    df = df.fillna('')
    df = df.astype(str)
    self.dataframe = df

  def preprocess_texts(self, quiet=False):
    self.dataframe['cleaned'] = preprocess(self.dataframe[self.text_col], quiet)
