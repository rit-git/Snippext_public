import numpy as np
import torch
import random
import jsonlines

from torch.utils import data
from .augment import Augmenter

tokenizer = None

def get_tokenizer(lm='bert'):
    """Return the tokenizer. Intiailize it if not initialized.

    Args:
        lm (string): the name of the language model (bert, albert, or distilbert)
    Returns:
        BertTokenizer or DistilBertTokenizer or AlbertTokenizer
    """
    global tokenizer
    if tokenizer is None:
        if lm == 'bert':
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif lm == 'distilbert':
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        elif lm == 'albert':
            from transformers import AlbertTokenizer
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        elif lm == 'roberta':
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif lm == 'xlnet':
            from transformers import XLNetTokenizer
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        elif lm == 'longformer':
            from transformers import LongformerTokenizer
            tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    return tokenizer


class SnippextDataset(data.Dataset):
    def __init__(self,
                 source,
                 vocab,
                 taskname,
                 max_len=512,
                 lm='bert',
                 augment_index=None,
                 augment_op=None,
                 size=None):
        """ TODO
        Args:

        """
        # tokens and tags
        sents, tags_li = [], [] # list of lists
        self.max_len = max_len
        get_tokenizer(lm)

        if type(source) is str:
            # read from file (for training/evaluation)
            if '_tagging' in taskname or '_qa' in taskname:
                sents, tags_li = self.read_tagging_file(source)
            else:
                sents, tags_li = self.read_classification_file(source)
            if size is not None:
                sents, tags_li = sents[:size], tags_li[:size]
        else:
            # read from list of tokens (for prediction)
            if '_tagging' in taskname or '_qa' in taskname:
                for tokens in source:
                    sents.append(["[CLS]"] + [token for token in tokens] + ["[SEP]"])
                    tags_li.append(["<PAD>"] + ['O' for token in tokens] + ["<PAD>"])
            else:
                for sent in source:
                    sents.append(sent)
                    tags_li.append(vocab[0])

        # handling QA datasets. Mark the question tokens with <PAD> so that
        # the model does not predict those tokens.
        if '_qa' in taskname:
            for tokens, labels in zip(sents, tags_li):
                if "[SEP]" in tokens[:-1]:
                    for i, token in enumerate(tokens):
                        labels[i] = "<PAD>"
                        if token == "[SEP]":
                            break

        # assign class variables
        self.sents, self.tags_li = sents, tags_li
        self.vocab = vocab

        # add special tags for tagging
        if '_tagging' in taskname:
            if 'O' not in self.vocab:
                self.vocab.append('O')
            if self.vocab[0] != '<PAD>':
                self.vocab.insert(0, '<PAD>')

        # index for tags/labels
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.vocab)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.vocab)}
        self.taskname = taskname

        # augmentation index and op
        if augment_op == 't5':
            self.augmenter = None
            self.augment_op = augment_op
            # read augmented examples
            self.augmented_examples = []
            if '_tagging' in taskname:
                with jsonlines.open(source + '.augment.jsonl', mode='r') as reader:
                    for row in reader:
                        exms = []
                        for entry in row['augment']:
                            tokens, labels = self.read_tagging_file(entry, is_file=False)
                            exms.append((tokens[0], labels[0]))
                        self.augmented_examples.append(exms)
            else:
                with jsonlines.open(source + '.augment.jsonl', mode='r') as reader:
                    for row in reader:
                        exms = []
                        label = row['label']
                        for entry in row['augment']:
                            sent = ' [SEP] '.join(entry.split('\t'))
                            exms.append((sent, label))
                        self.augmented_examples.append(exms)
        elif augment_index != None:
            self.augmenter = Augmenter(augment_index)
            self.augment_op = augment_op
        else:
            self.augmenter = None
            self.augment_op = None

    def read_tagging_file(self, path, is_file=True):
        """Read a train/eval tagging dataset from file

        The input file should contain multiple entries separated by empty lines.
        The format of each entry:

        The O
        room B-AS
        is O
        very B-OP
        clean I-OP
        . O

        Args:
            path (str): the path to the dataset file

        Returns:
            list of list of str: the tokens
            list of list of str: the labels
        """
        sents, tags_li = [], []
        if is_file:
            entries = open(path, 'r').read().strip().split("\n\n")
        else:
            entries = [path.strip()]

        for entry in entries:
            try:
                words = [line.split()[0] for line in entry.splitlines()]
                tags = [line.split()[-1] for line in entry.splitlines()]
                sents.append(["[CLS]"] + words[:self.max_len] + ["[SEP]"])
                tags_li.append(["<PAD>"] + tags[:self.max_len] + ["<PAD>"])
            except:
                print('error @', entry)
        return sents, tags_li


    def read_classification_file(self, path):
        """Read a train/eval classification dataset from file

        The input file should contain multiple lines where each line is an example.
        The format of each line:
        The room is clean.\troom\tpositive

        Args:
            path (str): the path to the dataset file

        Returns:
            list of str: the input sequences
            list of str: the labels
        """
        sents, labels = [], []
        lines = open(path).readlines()
        for line in lines:
            items = line.strip().split('\t')
            # only consider sentence and sentence pairs
            if len(items) < 2 or len(items) > 3:
                continue
            try:
                if len(items) == 2:
                    sents.append(items[0])
                    labels.append(items[1])
                else:
                    sents.append(items[0] + ' [SEP] ' + items[1])
                    labels.append(items[2])
            except:
                print('error @', line.strip())
        return sents, labels


    def __len__(self):
        """Return the length of the dataset"""
        return len(self.sents)

    def get(self, idx, op=[]):
        ag = self.augmenter
        self.augmenter = None
        item = self.__getitem__(idx)
        self.augmenter = ag
        return item

    def __getitem__(self, idx):
        """Return the ith item of in the dataset.

        Args:
            idx (int): the element index
        Returns (TODO):
            words, x, is_heads, tags, mask, y, seqlen, self.taskname
        """
        words, tags = self.sents[idx], self.tags_li[idx]

        if '_tagging' in self.taskname:
            # apply data augmentation if specified
            if self.augment_op == 't5':
                if len(self.augmented_examples[idx]) > 0:
                    words, tags = random.choice(self.augmented_examples[idx])
            elif self.augmenter != None:
                words, tags = self.augmenter.augment(words, tags, self.augment_op)

            # We give credits only to the first piece.
            model_input = {}
            x, y = [], [] # list of ids
            is_heads = [] # list. 1: the token is the first piece of a word
            attention_mask = []
            token_type_ids = []

            special_tokens = set(tokenizer.special_tokens_map.values())
            for w, t in zip(words, tags):
                # avoid bad tokens
                w = w[:50]
                tokens = tokenizer.tokenize(w) if w not in special_tokens else [w]
                xx = tokenizer.convert_tokens_to_ids(tokens)
                if len(xx) == 0:
                    continue

                is_head = [1] + [0]*(len(tokens) - 1)

                t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
                yy = [self.tag2idx[each] for each in t]  # (T,)

                is_heads += is_head
                x += xx
                y += yy
                if w in special_tokens:
                    attention_mask += [0] * len(tokens)
                else:
                    attention_mask += [1] * len(tokens)
                token_type_ids += [0] * len(tokens)

                # make sure that the length of x is not too large
                if len(x) > self.max_len:
                    break

            # to string
            words = " ".join(words)
            tags = " ".join(tags)

            # construct the model's input
            model_input['input_ids'] = x
            model_input['attention_mask'] = attention_mask
            model_input['token_type_ids'] = token_type_ids
            model_input['y'] = y
            model_input['is_heads'] = is_heads
        else: # classification
            if self.augment_op == 't5':
              if len(self.augmented_examples[idx]) > 0:
                  words, tags = random.choice(self.augmented_examples[idx])
            elif self.augmenter != None:
                words = self.augmenter.augment_sent(words, self.augment_op)

            if ' [SEP] ' in words:
                sent_a, sent_b = words.split(' [SEP] ')
            else:
                sent_a, sent_b = words, None

            model_input = tokenizer.encode_plus(sent_a, text_pair=sent_b,
                    truncation="longest_first",
                    max_length=self.max_len,
                    add_special_tokens=True)

            model_input['is_heads'] = []
            model_input['y'] = self.tag2idx[tags]

        model_input['words'] = words
        model_input['labels'] = tags
        model_input['taskname'] = self.taskname
        return model_input

    @staticmethod
    def pad(batch):
        '''Pads to the longest sample

        Args:
            batch (list of Dict): a list of model inputs to be batched

        Returns:
            Dictionary: the batched inputs
        '''
        nopad = lambda x: [sample[x] for sample in batch]
        pad = lambda x, seqlen, val: \
              [sample[x] + [val] * (seqlen - len(sample[x])) \
               for sample in batch] # 0: <pad>

        max_len = max([len(e['input_ids']) for e in batch])

        name = batch[0]['taskname']
        res = {'taskname': name}
        for attr, val in zip(['input_ids', 'attention_mask', 'token_type_ids'],
                [0, 0, 1]):
            res[attr] = torch.LongTensor(pad(attr, max_len, val))

        for attr in ['is_heads', 'words', 'labels']:
            res[attr] = nopad(attr)

        if '_tagging' in name:
            y = pad('y', max_len, 0)
        else:
            y = nopad('y')

        if 'sts-b' in name:
            y = torch.Tensor(y)
        else:
            y = torch.LongTensor(y)
        res['y'] = y

        return res # words, f(x), f(is_heads), tags, y, seqlens, name
