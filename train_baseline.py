import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import argparse
import json

from torch.utils import data
from model import MultiTaskNet
from dataset import *
from train_util import *
from tensorboardX import SummaryWriter
from transformers import AdamW


def train(model, train_set, optimizer, batch_size=32):
    """Perfrom one epoch of the training process.

    Args:
        model (MultiTaskNet): the current model state
        train_set (SnippextDataset): the training dataset
        optimizer: the optimizer for training (e.g., Adam)
        batch_size (int, optional): the batch size

    Returns:
        None
    """
    iterator = data.DataLoader(dataset=train_set,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=1,
                               collate_fn=SnippextDataset.pad)

    tagging_criterion = nn.CrossEntropyLoss(ignore_index=0)
    classifier_criterion = nn.CrossEntropyLoss()

    model.train()
    for i, batch in enumerate(iterator):
        # for monitoring
        words, x, is_heads, tags, mask, y, seqlens, taskname = batch
        taskname = taskname[0]
        _y = y

        if 'tagging' in taskname:
            criterion = tagging_criterion
        else:
            criterion = classifier_criterion

        # forward
        optimizer.zero_grad()
        logits, y, _ = model(x, y, mask=mask, task=taskname)
        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        loss = criterion(logits, y)

        # back propagation
        loss.backward()
        optimizer.step()

        if i == 0:
            print("=====sanity check======")
            print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hotel_tagging")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--bert_path", type=str, default=None)

    hp = parser.parse_args()

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = 'baseline_task_%s_batch_size_%d_run_id_%d' % (task, hp.batch_size, hp.run_id)

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]
    config_list = [config]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    task_type = config['task_type']
    vocab = config['vocab']
    tasknames = [task]

    # load train/dev/test sets
    train_dataset = SnippextDataset(trainset, vocab, task,
                                   max_len=64)
    valid_dataset = SnippextDataset(validset, vocab, task)
    test_dataset = SnippextDataset(testset, vocab, task)
    padder = SnippextDataset.pad

    valid_iter = data.DataLoader(dataset=valid_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=test_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        model = MultiTaskNet(config_list, device,
                         hp.finetuning, bert_path=hp.bert_path)
    else:
        model = MultiTaskNet(config_list, device,
                         hp.finetuning, bert_path=hp.bert_path).cuda()
        model = nn.DataParallel(model)

    # optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    optimizer = AdamW(model.parameters(), lr = hp.lr)

    # create logging directory
    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)
    writer = SummaryWriter(log_dir=hp.logdir)

    # start training
    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs+1):
        train(model,
              train_dataset,
              optimizer,
              batch_size=hp.batch_size)

        print(f"=========eval at epoch={epoch}=========")
        dev_f1, test_f1 = eval_on_task(epoch,
                            model,
                            task,
                            valid_iter,
                            valid_dataset,
                            test_iter,
                            test_dataset,
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

