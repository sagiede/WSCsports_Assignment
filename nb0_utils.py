# Standard library imports
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Data science imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# NLP imports
import nltk
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import spacy
import pandas as pd
# Jupyter settings
# %matplotlib inline
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Display settings for full text
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

# Jupyter settings
# %matplotlib inline
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

print('utils.py loaded')

# spaCy's stopword set
nlp = spacy.load("en_core_web_sm")
STOPWORDS = nlp.Defaults.stop_words

# Broad temporal concept vocabulary
temporal_candidates = {
    # basic temporal connectives
    "after", "before", "until", "till", "since", "when", "while", "once", "then", "later", "earlier",
    "eventually", "soon", "previously", "recently", "now",
    # specific time references
    "today", "tomorrow", "yesterday", "tonight", "morning", "afternoon", "evening",
    "day", "week", "month", "year", "season", "period", "half", "quarter",
    # sequence/order terms
    "final", "first", "second", "third", "last", "next"
}

# Broad negation vocabulary
negation_candidates = {
    "no", "not", "n't", "never", "cannot", "can't", "nobody", "none", "nothing", "nowhere",
    "neither", "nor", "without", "minus"
}

# Intersections with spaCy's stopword list
TEMPORAL_STOPWORDS = sorted({w for w in STOPWORDS if w in temporal_candidates})
NEGATION_STOPWORDS = sorted({w for w in STOPWORDS if w in negation_candidates})

def preprocess_text(text):
    """Lower, lemmatize, remove punct/space"""
    return [t.lemma_.lower() for t in nlp(text, disable=["parser", "ner"]) if not (t.is_punct or t.is_space)]

def lemmatize(text, phrases_patterns):
    text_tokens = preprocess_text(text)
    result = text_tokens[:]
    for pattern in phrases_patterns:
        pattern_split = pattern.split('_')
        if len(pattern_split) > 1:
            for i in range(len(result) - len(pattern_split) + 1):
                if result[i:i+len(pattern_split)] == pattern_split:
                    result = result[:i] + [pattern] + result[i+len(pattern_split):]
                    break
    result = [t for t in result if not nlp.vocab[t].is_stop or t in TEMPORAL_STOPWORDS + NEGATION_STOPWORDS]
    return result


# todo extract to utils
temporal_candidates = {
    # basic temporal connectives
    "after", "before", "until", "till", "since", "when", "while", "once", "then", "later", "earlier",
    "eventually", "soon", "previously", "recently", "now",
    # specific time references
    "today", "tomorrow", "yesterday", "tonight", "morning", "afternoon", "evening",
    "day", "week", "month", "year", "season", "period", "half", "quarter",
    # sequence/order terms
    "final", "first", "second", "third", "last", "next"
}

# Broad negation vocabulary
negation_candidates = {
    "no", "not", "n't", "never", "cannot", "can't", "nobody", "none", "nothing", "nowhere",
    "neither", "nor", "without", "minus"
}