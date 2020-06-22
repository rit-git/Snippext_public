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
from transformers import AdamW, get_linear_schedule_with_warmup
from apex import amp

# criterion for tagging
tagging_criterion = nn.CrossEntropyLoss(ignore_index=0)

# criterion for classification
classifier_criterion = nn.CrossEntropyLoss()

# criterion for regression
regression_criterion = nn.MSELoss()

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
    if 'sts-b' in taskname:
        logits = logits.view(-1)
    else:
        logits = logits.view(-1, logits.shape[-1])

    aug_y = y[batch_size:]
    y = y[:batch_size]
    aug_y = y.view(-1)
    y = y.view(-1)

    # cross entropy
    if 'tagging' in taskname:
        criterion = tagging_criterion
    elif 'sts-b' in taskname:
        criterion = regression_criterion
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
          scheduler=None,
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
        if scheduler:
            scheduler.step()

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

    # learning rate scheduler
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)
    # create logging
    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)
    writer = SummaryWriter(log_dir=hp.logdir)

    # start training
    best_dev_f1 = best_test_f1 = 0.0
    epoch = 1
    while epoch <= hp.n_epochs:
        train(model,
              trainset,
              augmentset,
              optimizer,
              scheduler=scheduler,
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

        # skip the epochs with zero f1
        if dev_f1 > 1e-6:
            epoch += 1
            if hp.save_model:
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    torch.save(model.state_dict(), run_tag + '_dev.pt')
                if test_f1 > best_test_f1:
                    best_test_f1 = dev_f1
                    torch.save(model.state_dict(), run_tag + '_test.pt')

    writer.close()

