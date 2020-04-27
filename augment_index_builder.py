import os
import json
import random
import math
import numpy as np
import argparse
import torch

from augment_utils import *
from transformers import BertModel
from nltk.corpus import wordnet
from nltk.corpus.reader.sentiwordnet import SentiWordNetCorpusReader
from gensim.models import Word2Vec

from snippext.dataset import get_tokenizer


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
    def __init__(self, train_fn, idf_fn, w2v, task, bert_path, lm='bert'):
        if 'tagging' in task or 'qa' in task:
            self.tokens, self.labels = read_tagging_file(train_fn)
        else:
            self.sents = read_asc_file(train_fn)
            self.tokens = list(map(lambda x: x['token'], self.sents))

        idf_dict = json.load(open(idf_fn))
        self.w2v = w2v
        self.task = task
        self.index = {'token': dict(), 'span': dict()}
        self.all_spans = list()
        self.span_freqs = list()
        self.avg_senti = dict()
        if self.task == 'classification':
            # sentiment sensitive
            self.calc_senti_score()
        self.tokenizer = get_tokenizer(lm=lm)
        self.init_token_index(idf_dict)
        self.init_span_index(bert_path=bert_path)
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

    def init_span_index(self, sim_token='cls', sim_topk=100, bert_path=None):
        if bert_path is None:
            bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        else:
            model_state_dict = torch.load(bert_path)
            bert_model = BertModel.from_pretrained('bert-base-uncased',
                    state_dict=model_state_dict,
                    output_hidden_states=True)
        bert_model.eval()

        aspect_dict = dict()
        opinion_dict = dict()
        aspect_token_list = []
        opinion_token_list = []
        aspect_raw_token_list = []
        opinion_raw_token_list = []

        n = len(self.tokens)
        max_len_as = 0
        max_len_op = 0
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
                opinions = []
                m = len(self.tokens[j])
                k = 0
                while k < m:
                    if 'B-AS' in self.labels[j][k]:
                        aspects.append([self.tokens[j][k]])
                        k += 1
                    elif 'I-AS' in self.labels[j][k]:
                        # ignore spans that are incorrectly labeled
                        if len(aspects) > 0:
                            aspects[-1].append(self.tokens[j][k])
                        k += 1
                    elif 'B-OP' in self.labels[j][k]:
                        opinions.append([self.tokens[j][k]])
                        k += 1
                    elif 'I-OP' in self.labels[j][k]:
                        # ignore spans that are incorrectly labeled
                        if len(opinions) > 0:
                            opinions[-1].append(self.tokens[j][k])
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

                for op_term in opinions:
                    tokenized_op = []
                    op_labels = []
                    for idx_op, w in enumerate(op_term):
                        tokenized_w = self.tokenizer.tokenize(w)
                        for idx_t, t in enumerate(tokenized_w):
                            if idx_t == 0:
                                op_labels.append(3 if idx_op == 0 else 4)
                            else:
                                op_labels.append(0)
                        tokenized_op += tokenized_w
                    len_op = len(tokenized_op)
                    max_len_op = max(len_op, max_len_op)
                    op_str = ' '.join(tokenized_op)
                    op_raw = ' '.join(op_term)
                    if op_raw not in opinion_dict:
                        opinion_dict[op_raw] = {
                            'document_freq': 1,
                            'bert_token': tokenized_op,
                            'bert_length': len_op,
                            'bert_label': op_labels,
                            'similar_spans': [],
                            'similar_spans_length': []
                        }
                        opinion_token_list.append(tokenized_op)
                        opinion_raw_token_list.append(op_raw)
                    else:
                        opinion_dict[op_raw]['document_freq'] += 1

        as_ids = []
        op_ids = []
        # Pad to max length and convert to ids
        for as_term in aspect_token_list:
            tk_as = ['[CLS]'] + as_term + ['[SEP]'] + ['[PAD]' for k in range(max_len_as - len(as_term))]
            as_ids.append(self.tokenizer.convert_tokens_to_ids(tk_as))
        for op_term in opinion_token_list:
            tk_op = ['[CLS]'] + op_term + ['[SEP]'] + ['[PAD]' for k in range(max_len_op - len(op_term))]
            op_ids.append(self.tokenizer.convert_tokens_to_ids(tk_op))

        # migrated to transformers
        if len(aspect_token_list) > 0:
            X_as = torch.LongTensor(as_ids)
            as_encoded_layers = bert_model(X_as)[2]
            X_as = as_encoded_layers[-2].detach()

        if len(opinion_token_list) > 0:
            X_op = torch.LongTensor(op_ids)
            op_encoded_layers = bert_model(X_op)[2]
            X_op = op_encoded_layers[-2].detach()

        # Compute the dot-product between all pairs of spans
        for i in range(len(aspect_token_list)):
            if sim_token == 'all':
                # using all tokens
                q = X_as[i]
                z = q * X_as
                score = torch.sum(z, dim=(1,2)) / torch.tensor(
                    np.linalg.norm(q) * np.linalg.norm(X_as, axis=(1,2)))
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

        for i in range(len(opinion_token_list)):
            if sim_token == 'all':
                # using all tokens
                q = X_op[i]
                z = q * X_op
                score = torch.sum(z, dim=(1,2)) / torch.tensor(
                    np.linalg.norm(q) * np.linalg.norm(X_op, axis=(1,2)))
            elif sim_token == 'cls':
                # using the CLS token
                q = X_op[i][0]
                z = q * X_op[:, 0, :]
                score = torch.sum(z, dim=(1)) / torch.tensor(
                    np.linalg.norm(q) * np.linalg.norm(X_op[:, 0, :], axis=(1)))
            elif sim_token == 'bas':
                # using the first token of the span
                q = X_op[i][1]
                z = q * X_op[:, 1, :]
                score = torch.sum(z, dim=(1)) / torch.tensor(
                    np.linalg.norm(q) * np.linalg.norm(X_op[:, 1, :], axis=(1)))
            topk_idx = torch.argsort(score, dim=0, descending=True)
            for idx in topk_idx:
                if idx == i:
                    continue
                if len(opinion_dict[opinion_raw_token_list[i]]['similar_spans']) < sim_topk:
                    opinion_dict[opinion_raw_token_list[i]]['similar_spans'].append(
                        [opinion_raw_token_list[idx], score[idx].item()])
                    opinion_dict[opinion_raw_token_list[i]]['similar_spans_length'].append(
                        opinion_dict[opinion_raw_token_list[idx]]['bert_length'])
                else:
                    break
        self.index['span'] =  {'aspect': aspect_dict, 'opinion': opinion_dict}


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
                        s_arr = s[0].split('_')
                        if s_arr[0] not in similar_words_dict:
                            similar_words_dict[s_arr[0]] = True
                        else:
                            continue
                        tokenized_s = self.tokenizer.tokenize(s_arr[0])
                        l_s = len(tokenized_s)
                        self.index['token'][w]['similar_words'].append([s_arr[0], s[1]])
                        # self.index['token'][w]['similar_words_bert'].append(tokenized_s)
                        self.index['token'][w]['similar_words_length'].append(l_s)

    def find_word_replacement(self, word_str, sim_topk=10, is_senti_sensitive=False):
        # find sim_topk similar words to word_str
        if is_senti_sensitive:
            # if sentiment sensitive, compute the senti score of word_str
            senti_score = 0
            word_str = word_str.lower()
            if word_str.lower() in self.avg_senti:
                senti_score = self.avg_senti[word_str.lower()]['pos_score'] - avg_senti[word_str.lower()]['neg_score']

        if self.w2v is None:
            # if Word2Vec is not given, using wordnet
            syns = wordnet.synsets(word_str)
            syn_list = []
            for syn in syns:
                for lem in syn.lemmas():
                    if lem.name() != word_str:
                        if is_senti_sensitive:
                            lem_senti_score = 0
                            if lem_str in self.avg_senti:
                                lem_senti_score = self.avg_senti[lem_str]['pos_score'] - self.avg_senti[lem_str]['neg_score']
                            ''' maybe we can use a different way to determine whether two words
                            are of the same sentiment '''
                            if sign(lem_senti_score) == sign(senti_score):
                                syn_list.append(lem_str)
                        else:
                            syn_list.append(lem.name())
            if len(syn_list) == 0:
                return []
            return list(zip(syn_list, [1.0 for i in range(len(syn_list))]))
        else:
            if word_str in self.w2v.wv.vocab:
                # if word_str appears in Word2Vec vocabulary
                similar_list = self.w2v.wv.most_similar(positive=[word_str], topn=sim_topk)
                if is_senti_sensitive:
                    arr = []
                    for ws in similar_list:
                        w_senti_score = 0
                        w = ws[0]
                        if w in self.avg_senti:
                            w_senti_score = self.avg_senti[w]['pos_score'] - self.avg_senti[w]['neg_score']
                        if sign(w_senti_score) == sign(senti_score):
                            arr.append(ws)
                    return arr
                else:
                    return similar_list
            else:
                # if word_str does not appear in Word2Vec vocabulary, find a synonym of it using WordNet
                # if the synonym appears in Word2Vec vocabulary, use similar words of this synonym
                syns = wordnet.synsets(word_str)
                syns_dict = dict()
                arr = []
                for syn in syns:
                    flag = False
                    for lem in syn.lemmas():
                        if lem.name() != word_str:
                            syns_dict[lem.name()] = True
                            if lem.name() in self.w2v.wv.vocab:
                                similar_list = self.w2v.wv.most_similar(positive=[lem.name()], topn=sim_topk)
                                if is_senti_sensitive:
                                    for ws in similar_list:
                                        w_senti_score = 0
                                        w = ws[0]
                                        if w in self.avg_senti:
                                            w_senti_score = self.avg_senti[w]['pos_score'] - self.avg_senti[w]['neg_score']
                                        if sign(w_senti_score) == sign(senti_score):
                                            arr.append(ws)
                                else:
                                    arr = similar_list
                                flag = True
                                break
                    if flag:
                        break
                if len(arr) == 0:
                    res = list(syns_dict.keys())
                    return list(zip(res, [1.0 for i in range(len(res))]))
                else:
                    return arr

    def calc_senti_score(self, swn_filename='combined_data/SentiWordNet_3.0.0_20100705.txt'):
        # aggregate sentiment score of tokens using SentiWordNet
        swn = SentiWordNetCorpusReader('./', [swn_filename])
        for senti_synset in swn.all_senti_synsets():
            w = senti_synset.synset.name().split('.')[0]
            if w not in self.avg_senti:
                self.avg_senti[w] = {
                    'pos_score': 0,
                    'neg_score': 0,
                    'count': 0
                }
            self.avg_senti[w]['pos_score'] += senti_synset.pos_score()
            self.avg_senti[w]['neg_score'] += senti_synset.neg_score()
            self.avg_senti[w]['count'] += 1

        for w in self.avg_senti:
            self.avg_senti[w]['pos_score'] /= self.avg_senti[w]['count']
            self.avg_senti[w]['neg_score'] /= self.avg_senti[w]['count']

    def dump_index(self, index_filename='augment_index.json'):
        outfile = open(index_filename, 'w')
        json.dump(self.index, outfile)
        outfile.close()


# def build_idf_dict(text_path):
#     from gensim.utils import simple_preprocess
#     from collections import Counter
#
#     cnt = Counter()
#     N = 0
#     for line in open(text_path):
#         tokens = simple_preprocess(line.lower())
#         tokens = set(tokens)
#         if len(tokens) > 0:
#             N += 1
#             for token in tokens:
#                 cnt[token] += 1
#
#     idf_dict = {}
#     for token in cnt:
#         idf_dict[token] = math.log(N / cnt[token])
#     return idf_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hotel_tagging")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--w2v_path", type=str, default="../rest_w2v.model")
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--lm", type=str, default='bert')
    parser.add_argument("--idf_path", type=str, default=None)
    parser.add_argument("--index_output_path", type=str, default="augment_index.json")

    hp = parser.parse_args()
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[hp.task]
    if hp.train_path is None:
        train_fn = config['trainset']
    else:
        train_fn = hp.train_path
    w2v = Word2Vec.load(hp.w2v_path)
    if hp.idf_path[-5:] != '.json':
        idf_fn = hp.idf_path + '.json'
        if not os.path.exists(idf_fn):
            idf_dict = build_idf_dict(hp.idf_path)
            json.dump(idf_dict, open(idf_fn, 'w'))
    else:
        idf_fn = hp.idf_path

    ib = IndexBuilder(train_fn, idf_fn, w2v,
                      config['task_type'],
                      hp.bert_path,
                      lm=hp.lm)
    ib.dump_index(hp.index_output_path)
