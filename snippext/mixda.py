import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import argparse
import json
import copy
import random

from torch.utils import data
from .model import MultiTaskNet
from .train_util import *
from .dataset import *
from tensorboardX import SummaryWriter
from transformers import AdamW
from apex import amp

# criterion for tagging
tagging_criterion = nn.CrossEntropyLoss(ignore_index=0)

# criterion for classification
classifier_criterion = nn.CrossEntropyLoss()

def mixda(model, batch, alpha_aug=0.4):
    """Perform one iteration of MixDA

    Args:
        model (MultiTaskNet): the model state
        batch (tuple): the input batch
        alpha_aug (float, Optional): the parameter for MixDA

    Returns:
        Tensor: the loss (of 0-d)
    """
    _, x, _, _, mask, y, _, taskname = batch
    taskname = taskname[0]
    # two batches
    batch_size = x.size()[0] // 2

    # augmented
    aug_x = x[batch_size:]
    aug_y = y[batch_size:]
    aug_lam = np.random.beta(alpha_aug, alpha_aug)

    # labeled
    x = x[:batch_size]

    # back prop
    logits, y, _ = model(x, y,
                         augment_batch=(aug_x, aug_lam),
                         task=taskname)
    logits = logits.view(-1, logits.shape[-1])

    aug_y = y[batch_size:]
    y = y[:batch_size]
    aug_y = y.view(-1)
    y = y.view(-1)

    # cross entropy
    if 'tagging' in taskname:
        criterion = tagging_criterion
    else:
        criterion = classifier_criterion

    # mix the labels
    loss = criterion(logits, y) * aug_lam + \
           criterion(logits, aug_y) * (1 - aug_lam)

    return loss


def create_mixda_batches(l_set, aug_set, batch_size=16):
    """Create batches for mixda

    Each batch is the concatenation of (1) a labeled batch and (2) an augmented
    labeled batch (having the same order of (1) )

    Args:
        l_set (SnippextDataset): the train set
        aug_set (SnippextDataset): the augmented train set
        batch_size (int, optional): batch size (of each component)

    Returns:
        list of list: the created batches
    """
    mixed_batches = []
    num_labeled = len(l_set)
    l_index = np.random.permutation(num_labeled)

    l_batch = []
    l_batch_aug = []
    padder = l_set.pad

    for i, idx in enumerate(l_index):
        l_batch.append(l_set[idx])
        l_batch_aug.append(aug_set[idx])

        if len(l_batch) == batch_size or i == len(l_index) - 1:
            batches = l_batch + l_batch_aug
            mixed_batches.append(padder(batches))
            l_batch.clear()
            l_batch_aug.clear()

    random.shuffle(mixed_batches)
    return mixed_batches


def train(model, l_set, aug_set, optimizer,
          fp16=False,
          batch_size=32,
          alpha_aug=0.8):
    """Perform one epoch of MixDA

    Args:
        model (MultiTaskModel): the model state
        train_dataset (SnippextDataset): the train set
        augment_dataset (SnippextDataset): the augmented train set
        optimizer (Optimizer): Adam
        fp16 (boolean, Optional): whether to use fp16
        batch_size (int, Optional): batch size
        alpha_aug (float, Optional): the alpha for MixDA

    Returns:
        None
    """
    mixda_batches = create_mixda_batches(l_set,
                                         aug_set,
                                         batch_size=batch_size)

    model.train()
    for i, batch in enumerate(mixda_batches):
        # for monitoring
        words, x, is_heads, tags, mask, y, seqlens, taskname = batch
        taskname = taskname[0]
        _y = y

        # perform mixmatch
        optimizer.zero_grad()
        loss = mixda(model, batch, alpha_aug)
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        if i == 0:
            print("=====sanity check======")
            print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", get_tokenizer().convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            y_sample = _y.cpu().numpy()[0]
            if np.isscalar(y_sample):
                print("y:", y_sample)
            else:
                print("y:", y_sample[:seqlens[0]])
            print("tags:", tags[0])
            print("mask:", mask[0])
            print("seqlen:", seqlens[0])
            print("task_name:", taskname)
            print("=======================")

        if i%10 == 0: # monitoring
            print(f"step: {i}, task: {taskname}, loss: {loss.item()}")
            del loss



def initialize_and_train(task_config,
                         trainset,
                         augmentset,
                         validset,
                         testset,
                         hp,
                         run_tag):
    """The train process.

    Args:
        task_config (dictionary): the configuration of the task
        trainset (SnippextDataset): the training set
        augmentset (SnippextDataset): the augmented training set
        validset (SnippextDataset): the validation set
        testset (SnippextDataset): the testset
        hp (Namespace): the parsed hyperparameters
        run_tag (string): the tag of the run (for logging purpose)

    Returns:
        None
    """
    padder = SnippextDataset.pad

    # iterators for dev/test set
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)


    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        model = MultiTaskNet([task_config], device,
                         hp.finetuning, lm=hp.lm, bert_path=hp.bert_path)
        optimizer = AdamW(model.parameters(), lr=hp.lr)
    else:
        model = MultiTaskNet([task_config], device,
                         hp.finetuning, lm=hp.lm, bert_path=hp.bert_path).cuda()
        optimizer = AdamW(model.parameters(), lr=hp.lr)
        if hp.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # create logging
    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)
    writer = SummaryWriter(log_dir=hp.logdir)

    # start training
    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs+1):
        train(model,
              trainset,
              augmentset,
              optimizer,
              fp16=hp.fp16,
              batch_size=hp.batch_size,
              alpha_aug=hp.alpha_aug)

        print(f"=========eval at epoch={epoch}=========")
        dev_f1, test_f1 = eval_on_task(epoch,
                            model,
                            task_config['name'],
                            valid_iter,
                            validset,
                            test_iter,
                            testset,
                            writer,
                            run_tag)

        if hp.save_model:
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                torch.save(model.state_dict(), run_tag + '_dev.pt')
            if test_f1 > best_test_f1:
                best_test_f1 = dev_f1
                torch.save(model.state_dict(), run_tag + '_test.pt')

    writer.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hotel_tagging")
    parser.add_argument("--lm", type=str, default="bert")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--augment_index", type=str, default=None)
    parser.add_argument("--augment_op", type=str, default=None)

    hp = parser.parse_args()

    task = hp.task # consider a single task for now

    # import nvidia apex if fp16 is on
    if hp.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from "
                              "https://www.github.com/nvidia/apex"
                              " to use fp16 training.")

    # create the tag of the run
    run_tag = 'mixda_task_%s_lm_%s_batch_size_%d_alpha_aug_%.1f_augment_op_%s_run_id_%d' % \
        (task, hp.lm, hp.batch_size, hp.alpha_aug, hp.augment_op, hp.run_id)

    # task config
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    task_type = config['task_type']
    vocab = config['vocab']
    tasknames = [task]

    # train dataset
    train_dataset = SnippextDataset(trainset, vocab, task,
                                   lm=hp.lm,
                                   max_len=hp.max_len)
    # train dataset augmented
    augment_dataset = SnippextDataset(trainset, vocab, task,
                                      lm=hp.lm,
                                      max_len=hp.max_len,
                                      augment_index=hp.augment_index,
                                      augment_op=hp.augment_op)
    # dev set
    valid_dataset = SnippextDataset(validset, vocab, task, lm=hp.lm)

    # test set
    test_dataset = SnippextDataset(testset, vocab, task, lm=hp.lm)

    # run the training process
    initialize_and_train(config,
                         train_dataset,
                         augment_dataset,
                         valid_dataset,
                         test_dataset,
                         hp, run_tag)
