import argparse
import json

from snippext.baseline import initialize_and_train
from snippext.dataset import SnippextDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sentiment_analysis") # laptop_ae_tagging, sentiment_analysis
    parser.add_argument("--lm", type=str, default="bert")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true", default="true")
    parser.add_argument("--logdir", type=str, default="./checkpoints_sentiment_analysis/baseline") # /checkpoints_sentiment_analysis
    parser.add_argument("--bert_path", type=str, default=None)

    hp = parser.parse_args()

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = 'baseline_task_%s' % task
    print("Run tag ", run_tag)

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    task_type = config['task_type']
    vocab = config['vocab']
    tasknames = [task]

    # load train/dev/test sets
    train_dataset = SnippextDataset(trainset, vocab, task, lm=hp.lm, max_len=hp.max_len)
    print("Training size ", len(train_dataset.sents))
    valid_dataset = SnippextDataset(validset, vocab, task, lm=hp.lm)
    test_dataset = SnippextDataset(testset, vocab, task, lm=hp.lm)
    print("Testing size ", len(test_dataset.sents))

    # run the training process
    initialize_and_train(config, train_dataset, valid_dataset, test_dataset, hp, run_tag)
