from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# from dataset import tokenizer
from gensim.utils import simple_preprocess

stopword_set = set(stopwords.words('english'))

def read_asc_file(fn):
    res = []
    for line in open(fn):
        if len(line) < 3:
            continue
        else:
            LL = line.strip().split('\t')
            # token = twt.tokenize(LL[0])
            # term = twt.tokenize(LL[1])
            tokens = simple_preprocess(LL[0])
            term = LL[1]
            polarity = LL[2]
            res.append({
                'raw': line,
                'token': tokens,
                'term': term,
                'polarity': polarity
            })
    return res

def is_stopword(token):
    return token in ['[SEP]', '[CLS]'] or token in stopword_set or not token.isalpha()

def read_tagging_file(fn):
    tokens = [[]]
    labels = [[]]
    for line in open(fn):
        if len(line) < 3:
            tokens.append([])
            labels.append([])
        else:
            LL = line.strip().split(' ')
            token = LL[0]
            label = LL[-1]
            tokens[-1].append(token)
            labels[-1].append(label)
    return tokens, labels

def build_idf_dict(fn):
    corpus = ['']
    cnt = 0
    for line in open(fn):
        if len(line) < 2:
            corpus.append('')
            cnt+=1
            # if cnt % 10 == 0:
            #     print(cnt)
        else:
            corpus[-1] += ' ' + line

    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    idf_dict = dict()
    for w in vectorizer.vocabulary_:
        idf_dict[w] = vectorizer.idf_[vectorizer.vocabulary_[w]]
    return idf_dict

def sign(a):
    return (a > 0) - (a < 0)
