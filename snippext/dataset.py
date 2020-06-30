import numpy as np
import torch

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
                 augment_op=None):
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
        if augment_index != None:
            self.augmenter = Augmenter(augment_index)
            self.augment_op = augment_op
        else:
            self.augmenter = None


    def read_tagging_file(self, path):
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
        entries = open(path, 'r').read().strip().split("\n\n")
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
        for line in open(path):
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
            if self.augmenter != None:
                words, tags = self.augmenter.augment(words, tags, self.augment_op)

            # We give credits only to the first piece.
            x, y = [], [] # list of ids
            is_heads = [] # list. 1: the token is the first piece of a word

            for w, t in zip(words, tags):
                # avoid bad tokens
                w = w[:50]
                tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                xx = tokenizer.convert_tokens_to_ids(tokens)
                if len(xx) == 0:
                    continue

                is_head = [1] + [0]*(len(tokens) - 1)

                t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
                yy = [self.tag2idx[each] for each in t]  # (T,)

                x.extend(xx)
                is_heads.extend(is_head)
                y.extend(yy)
                # make sure that the length of x is not too large
                if len(x) > self.max_len:
                    break

            assert len(x)==len(y)==len(is_heads), \
              f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}, {' '.join(tokens)}"

            # seqlen
            seqlen = len(y)

            mask = [1] * seqlen
            # masking for QA
            for i, t in enumerate(tags):
                if t != '<PAD>':
                    break
                mask[i] = 0

            # to string
            words = " ".join(words)
            tags = " ".join(tags)
        else: # classification
            if self.augmenter != None:
                words = self.augmenter.augment_sent(words, self.augment_op)

            if ' [SEP] ' in words:
                sents = words.split(' [SEP] ')
                tokens = ["[CLS]"] + \
                         tokenizer.tokenize(sents[0]) + \
                         ['[SEP]'] + \
                         tokenizer.tokenize(sents[1]) + \
                         ["[SEP]"]
            else:
                tokens = ["[CLS]"] + tokenizer.tokenize(words) + ["[SEP]"]
            x = tokenizer.convert_tokens_to_ids(tokens)[:self.max_len]
            y = self.tag2idx[tags] # label
            is_heads = [1] * len(x)
            mask = [1] * len(x)

            assert len(x)==len(mask)==len(is_heads), \
              f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"
            # seqlen
            seqlen = len(mask)

        return words, x, is_heads, tags, mask, y, seqlen, self.taskname

    @staticmethod
    def pad(batch):
        '''Pads to the longest sample

        Args:
            batch:

        Returns (TODO):
            return words, f(x), is_heads, tags, f(mask), f(y), seqlens, name
        '''
        f = lambda x: [sample[x] for sample in batch]
        g = lambda x, seqlen, val: \
              [sample[x] + [val] * (seqlen - len(sample[x])) \
               for sample in batch] # 0: <pad>

        # get maximal sequence length
        seqlens = f(6)
        maxlen = np.array(seqlens).max()

        # get task name
        name = f(7)

        words = f(0)
        x = g(1, maxlen, 0)
        is_heads = f(2)
        tags = f(3)
        mask = g(4, maxlen, 1)
        if '_tagging' in name[0]:
            y = g(5, maxlen, 0)
        else:
            y = f(5)

        f = torch.LongTensor
        if isinstance(y[0], float):
            y = torch.Tensor(y)
        else:
            y = torch.LongTensor(y)
        return words, f(x), is_heads, tags, f(mask), y, seqlens, name
