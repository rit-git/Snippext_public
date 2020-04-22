import os
import argparse
import json

from torch.utils import data
from snippext.baseline import initialize_and_train
from snippext.dataset import SnippextDataset

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hotel_tagging")
    parser.add_argument("--lm", type=str, default="bert")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--bert_path", type=str, default=None)

    hp = parser.parse_args()

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = 'baseline_task_%s_lm_%s_batch_size_%d_run_id_%d' % (task,
            hp.lm,
            hp.batch_size,
            hp.run_id)

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    task_type = config['task_type']
    vocab = config['vocab']
    tasknames = [task]

    # load train/dev/test sets
    train_dataset = SnippextDataset(trainset, vocab, task,
                                    lm=hp.lm,
                                    max_len=hp.max_len)
    valid_dataset = SnippextDataset(validset, vocab, task,
                                    lm=hp.lm)
    test_dataset = SnippextDataset(testset, vocab, task,
                                   lm=hp.lm)

    # run the training process
    initialize_and_train(config,
                         train_dataset,
                         valid_dataset,
                         test_dataset,
                         hp,
                         run_tag)
