import os
import argparse
import json

from torch.utils import data
from snippext.dataset import SnippextDataset
from snippext.mixda import initialize_and_train


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
