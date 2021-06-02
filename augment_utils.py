import pandas as pd
# from dataset import tokenizer
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
stopword_set = set(stopwords.words('english'))


def read_asc_file(train_fn):
    res = []
    df = pd.read_csv(train_fn, names=['id', 'review', 'sentiment'])
    for index, row in df.iterrows():
        if row['review'] is not np.nan and len(row['review']) > 2:
            term = row['review']
            if row['sentiment'] == 0:
                polarity = 'negative'
            else:
                polarity = 'positive'
            tokens = simple_preprocess(row['review'])
            res.append({
                'raw': row['review'],
                'token': tokens,
                'term': term,
                'polarity': polarity
            })
    return res


def is_stopword(token):
    return token in ['[SEP]', '[CLS]'] or token in stopword_set or not token.isalpha()


def read_tagging_file(fn):
    df = pd.read_json(fn, orient='index')

    tokens = [[]]
    labels = [[]]
    for index, row in df.iterrows():
        tokens.append(row['sentence'])
        labels.append(row['label'])
    return tokens, labels


def build_idf_dict(task):
    corpus = ['']
    cnt = 0

    if '_tagging' in task:
        for line in open('data/sentiment_analysis/sa_all_to_train.csv'):
            if len(line) < 2:
                corpus.append('')
                cnt += 1
            else:
                corpus[-1] += ' ' + line
    else:
        df = pd.read_csv('data/sentiment_analysis/sa_all_to_train.csv', names=['id', 'review', 'sentiment'])
        for index, row in df.iterrows():
            if row['review'] is not pd.np.nan and len(row['review']) < 2:
                corpus.append('')
                cnt += 1
            else:
                try:
                    corpus[-1] += ' ' + row['review']
                except:
                    print("Error @", row['review'])
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    idf_dict = dict()
    for w in vectorizer.vocabulary_:
        idf_dict[w] = vectorizer.idf_[vectorizer.vocabulary_[w]]
    return idf_dict