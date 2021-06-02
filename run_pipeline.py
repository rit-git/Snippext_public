import argparse
import csv
import json
import os
import sys

import torch
from gensim.utils import simple_preprocess
from torch.utils import data

# nlp = SpacyParser.get_instance().get_model(Lang.RO)
from snippext.dataset import SnippextDataset
from snippext.model import PredictionModel

# from rb.core.lang import Lang
# from rb.parser.spacy_parser import SpacyParser

csv.field_size_limit(sys.maxsize)


# nlp = spacy.load('en_core_web_sm')

def do_classification(input, config, model):
    """Apply the classification model.

    Args:
        input (str): the input review
        config (dict): the model configuration
        model (PredictionModel): the model in pytorch

    Returns:
        sentiment: the overall sentiment
    """
    # load data and tokenization
    dataset = SnippextDataset([input],
                              config['vocab'],
                              config['name'],
                              lm=model.lm,
                              max_len=64)
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=1,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=SnippextDataset.pad)

    # prediction
    model.eval()
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        print('Classify ', input)
        for i, batch in enumerate(iterator):
            try:
                words, x, is_heads, tags, mask, y, seqlens, taskname = batch
                taskname = taskname[0]
                logits, _, y_hat = model(x, y, task=taskname)  # y_hat: (N, T)

                print("Probability ", logits.softmax(dim=-1).max(dim=-1)[0].cpu().numpy().tolist())
                Words.extend(words)
                Is_heads.extend(is_heads)
                Tags.extend(tags)
                Y.extend(y.numpy().tolist())
                Y_hat.extend(y_hat.cpu().numpy().tolist())
            except:
                print('error @', batch)


def do_tagging(text, config, model):
    """Apply the tagging model.

    Args:
        text (str): the input paragraph
        config (dict): the model configuration
        model (PredictionModel): the model in pytorch

    Returns:
        list of list of str: the tokens in each sentences
        list of list of int: each token's starting position in the original text
        list of list of str: the tags assigned to each token
    """
    # load data and tokenization
    source = []
    token_pos_list = []
    # print('Tokenize sentences')
    # for sent in sent_tokenizer(text):
    #     tokens = [token.text for token in sent]
    #     token_pos = [token.idx for token in sent]
    #     source.append(tokens)
    #     token_pos_list.append(token_pos)

    review = simple_preprocess(text)
    print(review)

    # ['Bateria', 'nu', 'tine', 'foarte', 'mult','!'], ['Are', 'baterie',',' ,'dar'  ,'nu' ,'are' ,'incarcator']
    dataset = SnippextDataset([['are', 'un', 'cpu', 'foarte', 'puternic', '!'],
                               ['mi', 'se', 'pare', 'ca', 'ecranul', 'arata', 'foarte', 'bine']],
                              config['vocab'],
                              config['name'],
                              lm=model.lm,
                              max_len=64)
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=32,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=SnippextDataset.pad)

    # prediction
    model.eval()
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        # print('Tagging')
        for i, batch in enumerate(iterator):
            try:
                words, x, is_heads, tags, mask, y, seqlens, taskname = batch
                taskname = taskname[0]
                _, _, y_hat = model(x, y, task=taskname)  # y_hat: (N, T)

                Words.extend(words)
                Is_heads.extend(is_heads)
                Tags.extend(tags)
                Y.extend(y.numpy().tolist())
                Y_hat.extend(y_hat.cpu().numpy().tolist())
            except:
                print('error @', batch)

    # gets results and save
    results = []
    for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
        y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
        # remove the first and the last token
        preds = [dataset.idx2tag[hat] for hat in y_hat][1:-1]
        results.append(preds)

    print("Words ", words)
    print("Tagging ", results)


def do_pairing(all_tokens, all_tags, config, model):
    """Apply the pairing model.

    Args:
        all_tokens (list of list of str): the tokenized text
        all_tags (list of list of str): the tags assigned to each token
        config (dict): the model configuration
        model (PredictionModel): the model in pytorch

    Returns:
        list of dict: For each sentence, the list of extracted
            opinions/experiences from the sentence. Each dictionary includes
            an aspect term and an opinion term and the start/end
            position of the aspect/opinion term.
    """
    samples = []
    sent_ids = []
    candidates = []
    positions = []
    all_spans = {}

    sid = 0
    for tokens, tags in zip(all_tokens, all_tags):
        aspects = []
        opinions = []
        # find aspects
        # find opinions
        for i, tag in enumerate(tags):
            if tag[0] == 'B':
                start = i
                end = i
                while end + 1 < len(tags) and tags[end + 1][0] == 'I':
                    end += 1
                if tag == 'B-AS':
                    aspects.append((start, end))
                    all_spans[(sid, start, end)] = {'aspect': ' '.join(tokens[start:end + 1]),
                                                    'sid': sid,
                                                    'asp_start': start,
                                                    'asp_end': end}
                else:
                    opinions.append((start, end))
                    all_spans[(sid, start, end)] = {'opinion': ' '.join(tokens[start:end + 1]),
                                                    'sid': sid,
                                                    'op_start': start,
                                                    'op_end': end}

        candidate_pairs = []
        for asp in aspects:
            for opi in opinions:
                candidate_pairs.append((asp, opi))
        candidate_pairs.sort(key=lambda ao: abs(ao[0][0] - ao[1][0]))

        for asp, opi in candidate_pairs:
            asp_start, asp_end = asp
            op_start, op_end = opi
            token_ids = []
            for i in range(asp_start, asp_end + 1):
                token_ids.append((sid, i))
            for i in range(op_start, op_end + 1):
                token_ids.append((sid, i))

            if op_start < asp_start:
                samples.append(' '.join(tokens) + ' [SEP] ' + \
                               ' '.join(tokens[op_start:op_end + 1]) + ' ' + \
                               ' '.join(tokens[asp_start:asp_end + 1]))
            else:
                samples.append(' '.join(tokens) + ' [SEP] ' + \
                               ' '.join(tokens[asp_start:asp_end + 1]) + ' ' + \
                               ' '.join(tokens[op_start:op_end + 1]))

            sent_ids.append(sid)
            candidates.append({'opinion': ' '.join(tokens[op_start:op_end + 1]),
                               'aspect': ' '.join(tokens[asp_start:asp_end + 1]),
                               'sid': sid,
                               'asp_start': asp_start,
                               'asp_end': asp_end,
                               'op_start': op_start,
                               'op_end': op_end})
            positions.append(token_ids)
        sid += 1

    dataset = SnippextDataset(samples, config['vocab'], config['name'],
                              lm=model.lm)
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=32,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=SnippextDataset.pad)

    # prediction
    Y_hat = []
    Y = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, mask, y, seqlens, taskname = batch
            taskname = taskname[0]
            _, y, y_hat = model(x, y, task=taskname)  # y_hat: (N, T)
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            Y.extend(y.cpu().numpy().tolist())

    results = []
    for tokens in all_tokens:
        results.append({'sentence': ' '.join(tokens),
                        'extractions': []})

    used = set([])
    for i, yhat in enumerate(Y_hat):
        phrase = samples[i].split(' [SEP] ')[1]
        # print(phrase, yhat)
        if yhat == 1:
            # do some filtering
            assigned = False
            for tid in positions[i]:
                if tid in used:
                    assigned = True
                    break

            if not assigned:
                results[sent_ids[i]]['extractions'].append(candidates[i])
                for tid in positions[i]:
                    used.add(tid)
                # drop from all_spans
                sid = candidates[i]['sid']
                del all_spans[(sid,
                               candidates[i]['asp_start'],
                               candidates[i]['asp_end'])]
                del all_spans[(sid,
                               candidates[i]['op_start'],
                               candidates[i]['op_end'])]

    # add aspects/opinions that are not paired
    for sid, start, end in all_spans:
        results[sid]['extractions'].append(all_spans[(sid, start, end)])

    return results


def classify(extractions, config, model, sents=None):
    """Apply the classification models (for Sentiment and Attribute Classification).

    Args:
        extractions (list of dict): the partial extraction results by the pairing model
        config (dict): the model configuration
        model (PredictionModel): the model in pytorch

    Returns:
        list of dict: the extraction results with attribute name and sentiment score
            assigned to the field "attribute" and "sentiment".
    """
    phrases = []
    index = []
    # print('Prepare classification data')
    for sid, sent in enumerate(extractions):
        for eid, ext in enumerate(sent['extractions']):
            if 'asc' in config['name']:
                if 'aspect' in ext:
                    phrase = ' '.join(sents[ext['sid']]) + '\t' + ext['aspect']
                else:
                    phrase = ' '.join(sents[ext['sid']]) + '\t' + ext['opinion']
            else:
                if 'aspect' in ext and 'opinion' in ext:
                    phrase = ext['opinion'] + ' ' + ext['aspect']
                elif 'aspect' in ext:
                    phrase = ext['aspect']
                else:
                    phrase = ext['opinion']
            phrases.append(phrase)
            index.append((sid, eid))

    dataset = SnippextDataset(phrases, config['vocab'], config['name'],
                              lm=model.lm)
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=32,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=SnippextDataset.pad)

    # prediction
    Y_hat = []
    with torch.no_grad():
        # print('Classification')
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, mask, y, seqlens, taskname = batch
            taskname = taskname[0]
            _, _, y_hat = model(x, y, task=taskname)  # y_hat: (N, T)
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    for i in range(len(phrases)):
        attr = dataset.idx2tag[Y_hat[i]]
        sid, eid = index[i]
        if 'asc' in config['name']:
            extractions[sid]['extractions'][eid]['sentiment'] = attr
        else:
            extractions[sid]['extractions'][eid]['attribute'] = attr

    return extractions


def extract(review, config_list, models):
    """Extract experiences and opinions from a paragraph of text

    Args:
        review (Dictionary): a review object with a text field to be extracted
        config_list (list of dictionary): a list of task config dictionary
        models (list of PredictionModel): the most of models

    Returns:
        Dictionary: the same review object with a new extraction field
    """
    text = review['content']
    do_tagging(text, config_list[0], models[0])


def load_model(config,
               path,
               device='gpu',
               lm='readerbench/RoBERT-small'):
    """Load a model for a specific task.

    Args:
        config (dictionary): the task dictionary
        path (string): the path to the checkpoint
        lm (str, optional): the language model (bert, distilbert, or albert)
        fp16 (boolean): whether to use fp16 optimization

    Returns:
        PredictionModel: the model
    """
    if '_tagging' in config['name']:
        model = PredictionModel([config], device, True, lm=lm, task_type='laptop_ae_tagging')
        saved_state = torch.load('mixmatchnl_task_laptop_ae_tagging_test.pt', map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state)
        model = model.to(device)
    else:
        model = PredictionModel([config], device, True, lm=lm, task_type=config['task_type'])
        saved_state = torch.load('mixmatchnl_task_sentiment_analysis_test.pt',
                                 map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state)
        model = model.to(device)
    return model


def predict(config_list, models, task):
    """Run the extraction on an input csv file.

    Args:
        input_fn (str): the input file name (.csv)
        output_fn (str): the output file name (.jsonl)
        config_list (list of dict): the list of configuration
        models (list of PredictionModel): the list of models

    Returns:
        None
    """
    if '_tagging' in task:
        extract({'content': 'test'}, config_list, models)
    else:
        do_classification('Nu imi place acest produs, am dat banii degeaba!', config_list[0], models[0])
        do_classification('Merita toti bani, este genial!', config_list[0], models[0])
        do_classification('Este oribil, nu functioneaza!!!', config_list[0], models[0])


def initialize(checkpoint_path, tasks):
    """load the models from a path storing the checkpoints.

    Args:
        checkpoint_path (str): the path to the checkpoints
        use_gpu (boolean, optional): whether to use gpu
        lm (string, optional): the language model (default: bert)
        fp16 (boolean): whether to use fp16
        tasks (list of str, optional): the list of snippext tasks
    Returns:
        list of dictionary: the configuration list
        list of MultiTaskNet: the list of models
    """
    # load models
    checkpoints = dict([(task, os.path.join(checkpoint_path, \
                                            '%s.pt' % task)) for task in tasks])
    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = [load_model(configs[task], checkpoints[task], device=device) for task in tasks]
    config_list = [configs[task] for task in tasks]

    return config_list, models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints')
    parser.add_argument("--task", type=str, default='laptop_ae_tagging')  # laptop_ae_tagging
    hp = parser.parse_args()

    config_list, models = initialize(hp.checkpoint_path, [hp.task])
    predict(config_list, models, hp.task)
