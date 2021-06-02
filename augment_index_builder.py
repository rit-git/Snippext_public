import argparse
import json
import math
import os

import fasttext.util
import fasttext.util
import torch
from transformers import BertModel

from augment_utils import *
from snippext.dataset import get_tokenizer


def reverse_list_elements(my_list):
    other_list = []
    for inner_list in my_list:
        other_list.append([inner_list[1], inner_list[0]])
    return other_list


class IndexBuilder(object):
    """Index builder for span-level and token-level data augmentation

    Support both token and span level augmentation operators.

    Attributes:
        tokens (list of lists): tokens of training data
        labels (list of lists): labels of each token of training data (for AOE task)
        sents (list of dicts): training examples for ASC task
        w2v: Word2Vec model for similar word replacement
        index (dict): a dictionary containing both the token and span level index
        all_spans (list of str): a list of all the spans in the index
        span_freqs (list of int): the document frequency of each span in all_spans
        lm (string): the language model; 'bert' by default
    """

    def __init__(self, train_fn, idf_fn, ft, task):
        if 'tagging' in task or 'qa' in task:
            self.tokens, self.labels = read_tagging_file(train_fn)
        else:
            self.sents = read_asc_file(train_fn)
            self.tokens = list(map(lambda x: x['token'], self.sents))

        idf_dict = json.load(open(idf_fn))
        self.task = task
        self.ft = ft
        self.index = {'token': dict(), 'span': dict()}
        self.all_spans = list()
        self.span_freqs = list()
        self.avg_senti = dict()
        self.tokenizer = get_tokenizer()
        self.init_token_index(idf_dict)
        self.init_span_index()
        self.index_token_replacement()

    def init_token_index(self, idf_dict):
        oov_th = math.log(1e8)
        for token in self.tokens:
            for w in token:
                if w not in self.index['token']:
                    self.index['token'][w] = dict()
                    wl = w.lower()

                    if wl not in idf_dict:
                        self.index['token'][w]['idf'] = oov_th
                    else:
                        self.index['token'][w]['idf'] = idf_dict[wl]
                    tokenized_w = self.tokenizer.tokenize(w)
                    self.index['token'][w]['bert_token'] = tokenized_w
                    self.index['token'][w]['bert_length'] = len(tokenized_w)
                    self.index['token'][w]['similar_words'] = None
                    self.index['token'][w]['similar_words_bert'] = None
                    self.index['token'][w]['similar_words_length'] = None

    def init_span_index(self, sim_token='cls', sim_topk=100):
        bert_model = BertModel.from_pretrained('bert', output_hidden_states=True)
        bert_model.eval()

        aspect_dict = dict()
        aspect_token_list = []
        aspect_raw_token_list = []

        n = len(self.tokens)
        max_len_as = 0
        for j in range(n):
            if 'classification' in self.task:
                # for ASC datasets, the aspect is given in the term field
                as_term = self.sents[j]['term']

                # as_labels
                as_labels = []
                tokenized_as = self.tokenizer.tokenize(as_term)
                for idx_t, t in enumerate(tokenized_as):
                    if idx_t == 0:
                        as_labels.append(1 if idx_t == 0 else 2)
                    else:
                        as_labels.append(0)

                len_as = len(tokenized_as)
                max_len_as = max(len_as, max_len_as)
                if max_len_as > 512:
                    print("Here ", as_term)
                    raise Exception("Sorry, no numbers below zero")
                as_str = ' '.join(tokenized_as)
                if as_term not in aspect_dict:
                    aspect_dict[as_term] = {
                        'document_freq': 1,
                        'bert_token': tokenized_as,
                        'bert_length': len_as,
                        'bert_label': as_labels,
                        'similar_spans': [],
                        'similar_spans_length': []
                    }
                    aspect_token_list.append(tokenized_as)
                    aspect_raw_token_list.append(as_term)
                else:
                    aspect_dict[as_term]['document_freq'] += 1
            else:
                # for AOE datasets, we have to enumerate tokens to find all aspects and opinions
                aspects = []
                m = len(self.tokens[j])
                k = 0
                while k < m:
                    if 'B' in self.labels[j][k]:
                        aspects.append([self.tokens[j][k]])
                        k += 1
                    elif 'I' in self.labels[j][k]:
                        # ignore spans that are incorrectly labeled
                        if len(aspects) > 0:
                            aspects[-1].append(self.tokens[j][k])
                        k += 1
                    else:
                        k += 1

                for as_term in aspects:
                    tokenized_as = []
                    as_labels = []
                    for idx_as, w in enumerate(as_term):
                        tokenized_w = self.tokenizer.tokenize(w)
                        for idx_t, t in enumerate(tokenized_w):
                            if idx_t == 0:
                                as_labels.append(1 if idx_as == 0 else 2)
                            else:
                                as_labels.append(0)
                        tokenized_as += tokenized_w
                    len_as = len(tokenized_as)
                    max_len_as = max(len_as, max_len_as)
                    if max_len_as == 726:
                        print("Here")
                    as_str = ' '.join(tokenized_as)
                    as_raw = ' '.join(as_term)
                    if as_raw not in aspect_dict:
                        aspect_dict[as_raw] = {
                            'document_freq': 1,
                            'bert_token': tokenized_as,
                            'bert_length': len_as,
                            'bert_label': as_labels,
                            'similar_spans': [],
                            'similar_spans_length': []
                        }
                        aspect_token_list.append(tokenized_as)
                        aspect_raw_token_list.append(as_raw)
                    else:
                        aspect_dict[as_raw]['document_freq'] += 1

        as_ids = []
        # Pad to max length and convert to ids
        for as_term in aspect_token_list:
            tk_as = ['[CLS]'] + as_term + ['[SEP]'] + ['[PAD]' for k in range(max_len_as - len(as_term))]
            as_ids.append(self.tokenizer.convert_tokens_to_ids(tk_as))

        # migrated to transformers
        if len(aspect_token_list) > 0:
            X_as = torch.LongTensor(as_ids)
            as_encoded_layers = bert_model(X_as)[2]
            X_as = as_encoded_layers[-2].detach()

        # Compute the dot-product between all pairs of spans
        for i in range(len(aspect_token_list)):
            if sim_token == 'all':
                # using all tokens
                q = X_as[i]
                z = q * X_as
                score = torch.sum(z, dim=(1, 2)) / torch.tensor(
                    np.linalg.norm(q) * np.linalg.norm(X_as, axis=(1, 2)))
            elif sim_token == 'cls':
                # using the CLS token
                q = X_as[i][0]
                z = q * X_as[:, 0, :]
                score = torch.sum(z, dim=(1)) / torch.tensor(
                    np.linalg.norm(q) * np.linalg.norm(X_as[:, 0, :], axis=(1)))
            elif sim_token == 'bas':
                # using the first token of the span
                q = X_as[i][1]
                z = q * X_as[:, 1, :]
                score = torch.sum(z, dim=(1)) / torch.tensor(
                    np.linalg.norm(q) * np.linalg.norm(X_as[:, 1, :], axis=(1)))

            topk_idx = torch.argsort(score, dim=0, descending=True)
            for idx in topk_idx:
                if idx == i:
                    continue
                if len(aspect_dict[aspect_raw_token_list[i]]['similar_spans']) < sim_topk:
                    aspect_dict[aspect_raw_token_list[i]]['similar_spans'].append(
                        [aspect_raw_token_list[idx], score[idx].item()])
                    aspect_dict[aspect_raw_token_list[i]]['similar_spans_length'].append(
                        aspect_dict[aspect_raw_token_list[idx]]['bert_length'])
                else:
                    break
        self.index['span'] = {'aspect': aspect_dict}
        print("Finished init_span_index")

    def index_token_replacement(self):
        # pre-compute all token replacement candidates and store them in the index
        for token in self.tokens:
            for w in token:
                if is_stopword(w) or self.index['token'][w]['similar_words'] is not None:
                    continue
                self.index['token'][w]['similar_words'] = []
                # self.index['token'][w]['similar_words_bert'] = []
                self.index['token'][w]['similar_words_length'] = []

                synonyms = self.find_word_replacement(word_str=w)
                similar_words_dict = dict()
                if len(synonyms) >= 1:
                    for s in list(synonyms):
                        s_arr = str(s[0]).split('_')
                        if s_arr[0] not in similar_words_dict:
                            similar_words_dict[s_arr[0]] = True
                        else:
                            continue
                        tokenized_s = self.tokenizer.tokenize(s_arr[0])
                        l_s = len(tokenized_s)
                        self.index['token'][w]['similar_words'].append([s_arr[0], s[1]])
                        self.index['token'][w]['similar_words_length'].append(l_s)

    def find_word_replacement(self, word_str):
        # if word_str appears in Word2Vec vocabulary
        similar_list = self.ft.get_nearest_neighbors(word_str, k=10)
        if len(similar_list) == 0:
            print("No synonyms for {}", word_str)
        return reverse_list_elements(similar_list)

    def dump_index(self, index_filename='augment_index.json'):
        outfile = open(index_filename, 'w')
        json.dump(self.index, outfile)
        outfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sentiment_analysis")
    parser.add_argument("--idf_path", type=str, default='idf_sa.json')  # for tagging 'idf.json'
    parser.add_argument("--index_output_path", type=str,
                        default="augment_index_sa.json")  # for tagging "augment_index.json"

    ft = fasttext.load_model('./cc.ro.300.bin')
    fasttext.util.reduce_model(ft, 100)

    hp = parser.parse_args()
    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[hp.task]

    train_fn = config['trainset']

    idf_fn = hp.idf_path
    if not os.path.exists(idf_fn):
        idf_dict = build_idf_dict(hp.task)
        json.dump(idf_dict, open(idf_fn, 'w'))

    print("Train filename", train_fn)
    print("Idf path", idf_fn)
    print("Task type", config['task_type'])

    ib = IndexBuilder(train_fn, idf_fn, ft,
                      config['task_type'])

    ib.dump_index(hp.index_output_path)
