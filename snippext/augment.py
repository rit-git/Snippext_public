import json
import random
import numpy as np

class Augmenter(object):
    """Data augmentation for the extractor.

    Support both token and span level augmentation operators.

    Attributes:
        index (dict): a dictionary containing both the token and span level index
        all_spans (dict): a dictionary from span type to a list of all the spans in the index
        span_freqs (dict): a dictionary from span type to the document frequency of each span in all_spans
    """

    def __init__(self, index_fn, valid_spans=None):
        self.index = json.load(open(index_fn))
        self.all_spans = {}
        self.span_freqs = {}

        for span_type in self.index['span']:
            self.all_spans[span_type] = list(self.index['span'][span_type].keys())
            span_freqs_tmp = [self.index['span'][span_type][sp]['document_freq'] \
                               for sp in self.all_spans[span_type]]
            self.span_freqs[span_type] = np.array(span_freqs_tmp) / np.sum(span_freqs_tmp)

    def augment(self, tokens, labels, op='token_del_tfidf'):
        """ Performs data augmentation on a tagging example.

        We support deletion (del), insertion (ins), replacement (repl),
        and swapping(swap) at the token level. At the span level, we support
        replacement with the options of replacing with random, frequent (freq),
        or similar spans (sim).

        The supported ops:
          ['token_del_tfidf',
           'token_del',
           'token_repl_tfidf',
           'token_repl',
           'token_swap',
           'token_ins',
           'span_sim',
           'span_freq',
           'span']

        Args:
            tokens (list of strings): the input tokens
            labels (list of strings): the labels of the tokens
            op (str, optional): a string encoding of the operator to be applied

        Returns:
            list of strings: the augmented tokens
            list of strings: the augmented labels
        """
        flags = op.split('_')
        if 'span' in flags:
            start, end = self.sample_span_position(tokens, labels)
            if start < 0:
                return tokens, labels
            span = ' '.join(tokens[start:end+1])# .lower()
            label = labels[start]
            if label.startswith('B-'):
                labelI = 'I' + labels[start][1:]
            else:
                labelI = label

            if 'AS' in label:
                span_type = 'aspect'
            else:
                span_type = 'opinion'

            if 'sim' in op:
                candidates = self.index['span'][span_type][span]['similar_spans']
                new_span = random.choice(candidates)[0]
            elif 'freq' in op:
                candidates = self.all_spans[span_type]
                new_span = np.random.choice(candidates, 1,
                                            p=self.span_freqs[span_type])[0]
            else:
                candidates = self.all_spans[span_type]
                new_span = random.choice(candidates)

            new_span_len = len(new_span.split(' '))
            new_tokens = tokens[:start] + \
                    new_span.split(' ') + tokens[end+1:]
            new_labels = labels[:start] + [label] + \
                [labelI] * (new_span_len - 1) + labels[end+1:]
            return new_tokens, new_labels
        else:
            tfidf = 'tfidf' in op
            pos1 = self.sample_position(tokens, labels, tfidf)
            if pos1 < 0:
                return tokens, labels

            if 'del' in op:
                # insert padding to keep the length consistent
                if tokens[pos1] in self.index['token']:
                    length = self.index['token'][tokens[pos1]]['bert_length']
                else:
                    length = 1
                new_tokens = tokens[:pos1] + ['[PAD]']*(length) + tokens[pos1+1:]
                new_labels = labels[:pos1] + ['<PAD>']*(length) + labels[pos1+1:]
            elif 'ins' in op:
                ins_token = self.sample_token(tokens[pos1], same_length=False)
                new_tokens = tokens[:pos1] + [ins_token] + tokens[pos1:]
                new_labels = labels[:pos1] + ['O'] + labels[pos1:]
            elif 'repl' in op:
                ins_token = self.sample_token(tokens[pos1], same_length=False)
                if tokens[pos1] in self.index['token'] and \
                      ins_token in self.index['token']:
                    len1 = self.index['token'][tokens[pos1]]['bert_length']
                    len2 = self.index['token'][ins_token]['bert_length']
                    if len1 < len2:
                        # truncate the new sequence
                        bert_tokens = self.index['token'][ins_token]['bert_token'][:len1]
                        bert_tokens = [token.replace('##', '') for token in bert_tokens]
                        ins_token = ''.join(bert_tokens)
                        new_tokens = tokens[:pos1] + [ins_token] + tokens[pos1+1:]
                        new_labels = labels[:pos1] + ['O'] + labels[pos1+1:]
                    else:
                        # pad the new sequence
                        more = len1 - len2
                        new_tokens = tokens[:pos1] + [ins_token] + ['[PAD]']*more + tokens[pos1+1:]
                        new_labels = labels[:pos1] + ['O'] + ['<PAD>']*more + labels[pos1+1:]
                else:
                    # backup
                    new_tokens = tokens[:pos1] + [ins_token] + tokens[pos1+1:]
                    new_labels = labels[:pos1] + ['O'] + labels[pos1+1:]
            elif 'swap' in op:
                pos2 = self.sample_position(tokens, labels, tfidf)
                new_tokens = list(tokens)
                new_labels = list(labels)
                new_tokens[pos1], new_tokens[pos2] = tokens[pos2], tokens[pos1]
            else:
                new_tokens, new_labels = tokens, labels

            return new_tokens, new_labels


    def augment_sent(self, text, op='token_del_tfidf'):
        """ Performs data augmentation on a classification example.

        Similar to augment(tokens, labels) but works for sentences
        or sentence-pairs.

        Args:
            text (str): the input sentence
            op (str, optional): a string encoding of the operator to be applied

        Returns:
            str: the augmented sentence
        """
        # handling sentence pairs
        sents = text.split(' [SEP] ')
        text = sents[0]
        target_spans = sents[1:]

        # tokenize the sentence
        current = ''
        tokens = []
        labels = []
        for ch in text:
            if ch.isalnum():
                current += ch
            else:
                if current != '':
                    tokens.append(current)
                if ch not in ' \t\r\n':
                    tokens.append(ch)
                current = ''
        if current != '':
            tokens.append(current)

        labels = ['O'] * len(tokens)
        for idx, span in enumerate(target_spans):
            span_tokens = span.split(' ')
            for tid in range(len(tokens)):
                if tid + len(span_tokens) <= len(tokens) and \
                   tokens[tid:tid+len(span_tokens)] == span_tokens:
                    for i in range(tid, tid+len(span_tokens)):
                        labels[i] = 'SP%d' % idx

        # print(tokens)
        # print(labels)
        # only augment the original sentence
        tokens, labels = self.augment(tokens, labels, op=op)

        # check consistency
        tid = 0
        while tid < len(tokens):
            if labels[tid][:2] == 'SP':
                new_span = tokens[tid]
                idx = int(labels[tid][2:])
                while tid + 1 < len(tokens) and \
                      labels[tid + 1] == labels[tid]:
                    tid += 1
                    new_span += ' ' + tokens[tid]
                if target_spans[idx] != new_span:
                    target_spans[idx] = new_span
            tid += 1

        # error handling
        results = ' '.join(tokens)
        for span in target_spans:
            results += ' [SEP] ' + span
        return results


    def sample_position(self, tokens, labels, tfidf=False):
        """ Randomly sample a token's position from a training example

        When tfidf is turned on, the weight of each token is proportional
        to MaxTfIdf - Tfidf of each token. When it is off, the sampling is uniform.
        Only tokens with 'O' labels and at least 1 position away from a non 'O'
        labels will be sampled.

        Args:
            tokens (list of strings): the input tokens
            labels (list of strings): the labels of the tokens
            tfidf (bool, optional): whether the sampled position is by tfidf

        Returns:
            int: the sampled position (-1 if no such position)
        """
        index = self.index['token']
        candidates = []
        for idx, token in enumerate(tokens):
            if labels[idx] == 'O' and \
                   token in index and \
                   index[token]['similar_words'] != None and \
                   len(index[token]['similar_words']) > 0:
                candidates.append(idx)
            # if token.lower() in index and \
            #     labels[idx] == 'O' and \
            #     (idx + 1 >= len(tokens) or labels[idx + 1] == 'O') and \
            #     (idx - 1 < 0 or labels[idx - 1] == 'O'):
            #     candidates.append(idx)

        if len(candidates) <= 0:
            return -1
        if tfidf:
            weight = {}
            max_weight = 0.0
            for idx, token in enumerate(tokens):
                # token = token.lower()
                if token not in index:
                    continue
                if token not in weight:
                    weight[token] = 0.0
                weight[token] += index[token]['idf']
                max_weight = max(max_weight, weight[token])

            weights = []
            for idx in candidates:
                weights.append(max_weight - weight[tokens[idx]] + 1e-6)
                # weights.append(max_weight - weight[tokens[idx].lower()] + 1e-6)
            weights = np.array(weights) / sum(weights)

            return np.random.choice(candidates, 1, p=weights)[0]
        else:
            return random.choice(candidates)

    def sample_token(self, token, same_length=True, max_candidates=10):
        """ Randomly sample a token's similar token stored in the index

        Args:
            token (str): the input token
            same_length (bool, optional): whether the return token should have the same
                length in BERT
            max_candidates (int, optional): the maximal number of candidates
                to be sampled

        Returns:
            str: the sampled token (unchanged if the input is not in index)
        """
        # token = token.lower()
        index = self.index['token']
        if token in index and index[token]['similar_words'] != None:
            if same_length:
                bert_length = index[token]['bert_length']
                candidates = []
                for ts, bl in zip(index[token]['similar_words'],
                                  index[token]['similar_words_length']):
                    t, _ = ts
                    if bl == bert_length:
                        candidates.append(t)
                        if len(candidates) >= max_candidates:
                            break
            else:
                candidates = [t for t, _ in \
                              index[token]['similar_words'][:max_candidates]]
            if len(candidates) <= 0:
                return token
            else:
                return random.choice(candidates)
        else:
            return token

    def sample_span_position(self, tokens, labels):
        """ Uniformly sample a span from a training example and return its positions.

        The output is a pair (start_op, end_op) of the span.

        Args:
            tokens (list of strings): the input tokens
            labels (list of strings): the labels of the tokens

        Returns:
            int: the start position (-1 if no available span)
            int: the ending position (-1 if no available span)
        """
        index = self.index['span']
        candidates = []
        idx = 0
        while idx < len(tokens):
            if labels[idx] != 'O':
                start = idx
                while idx + 1 < len(tokens) and \
                    labels[idx + 1][1:] == labels[idx][1:]:
                    idx += 1
                end = idx

                span = ' '.join(tokens[start:end+1]) # .lower()
                if span in index:
                    candidates.append((start, end))
            idx += 1

        if len(candidates) > 0:
            return random.choice(candidates)
        else:
            return (-1, -1)


if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ag = Augmenter('augment/laptop_index.json', [])
    tokens = 'this is a great drawback desktop keyboard'.split(' ')
    labels = 'O O O O O B-AS I-AS'.split(' ')
    for op in ['token_del_tfidf',
               'token_del',
               'token_repl_tfidf',
               'token_repl',
               'token_swap',
               'token_ins',
               'span_sim',
               'span_freq',
               'span']:
        print(op)
        result = ag.augment(tokens, labels, op=op)
        result = ' '.join(result[0])
        print(result, len(tokenizer.encode(result)))
        original = ' '.join(tokens)
        print(original, len(tokenizer.encode(original)))

    tokens = 'I liked the macbook desktop keyboard . It is very good .'.split(' ')
    labels = 'O O O O B-AS I-AS O O O O O O'.split(' ')
    for op in ['span']:
        print(op)
        print(ag.augment(tokens, labels, op=op))


    ag = Augmenter('augment/rest_index_asc.json', [])
    text = 'I liked the beef [SEP] beef'
    for op in ['token_del_tfidf',
               'token_del',
               'token_repl_tfidf',
               'token_repl',
               'token_swap',
               'token_ins',
               'span_sim',
               'span_freq',
               'span']:
        print(op)
        print(ag.augment_sent(text, op=op))

    ag = Augmenter('augment/laptop_index_asc.json', [])
    text = 'I liked the desktop keyboard [SEP] desktop keyboard'
    for op in ['token_del_tfidf',
               'token_del',
               'token_repl_tfidf',
               'token_repl',
               'token_swap',
               'token_ins',
               'span_sim',
               'span_freq',
               'span']:
        print(op)
        print(ag.augment_sent(text, op=op))
