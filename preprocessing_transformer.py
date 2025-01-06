import numpy as np
import re
from joblib import dump
import pandas as pd

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig


# Dictionary of replacements
replacements = {
    "i" : "I",
    "im" : "I am",
    "Im" : "I am",
    "i m" : "I am",
    "I m" : "I am",
    "hadn t" : "had not",
    "hasn t" : "has not",
    "haven t" : "have not", 
    "don t" : "do not",
    "can t" : "cannot",
    "didn t" : "did not",
    "aren t" : "are not",
    "isn t" : "is not",
    "it s" : "it is",
    "ive" : "I have",
    "it d" : "it would",
    "how d" : "how did",
    "could ve" : "could have",
    "cuz" : "because",
    "gotta" : "got to",
    "kinda" : "kind of",
    "lemme" : "let me",
    "o clock" : "of the clock",
    "y ever" : "have you ever",
    "y know" : "you know",
    "you ll" : "you will",
    "you d" : "you had",
    "why s" : "why is",
    "why re" : "why are",
    "won t" : "will not",
    "would ve": "would have",
    "til" : "until",
    "tis" : "it is",
    "somebody s": "somebody is",
    "someone s": "someone is",
    "mine s" : "mine is"
}

# Function to apply replacements
def replace_words(text, replacements):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in replacements.keys()) + r')\b', re.IGNORECASE)
    return pattern.sub(lambda x: replacements[x.group().lower()], text)



def main():
    test = pd.read_csv('data/test.csv')
    train = pd.read_csv('data/training.csv')
    validation = pd.read_csv('data/validation.csv')

    # Remove the contractions in each DataFrame
    test['text'] = test['text'].apply(replace_words, replacements=replacements)
    train['text'] = train['text'].apply(replace_words, replacements=replacements)
    validation['text'] = validation['text'].apply(replace_words, replacements=replacements)

