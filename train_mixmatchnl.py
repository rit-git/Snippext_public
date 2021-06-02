import argparse
import json

from snippext.dataset import SnippextDataset
from snippext.mixmatchnl import initialize_and_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="laptop_ae_tagging") # laptop_ae_tagging, sentiment_analysis
    parser.add_argument("--lm", type=str, default="bert")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true", default="true")
    parser.add_argument("--logdir", type=str, default="./checkpoints")
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--num_aug", type=int, default=2)
    parser.add_argument("--u_lambda", type=float, default=10.0)
    parser.add_argument("--augment_index", type=str, default='augment_index.json') #augment_index_sa.json
    parser.add_argument("--augment_op", type=str, default='token_repl_tfidf')

    hp = parser.parse_args()
    task = hp.task

    # create the tag of the run
    run_tag = 'mixmatchnl_task_%s' % task

    # task config
    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[task]
    config_list = [config]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    unlabeled = config['unlabeled']
    task_type = config['task_type']
    vocab = config['vocab']

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

    # unlabeled dataset
    u_dataset = SnippextDataset(unlabeled, vocab, task, max_len=hp.max_len, lm=hp.lm)

    # augmented dataset
    u_dataset_aug = SnippextDataset(unlabeled, vocab, task,
                                    lm=hp.lm,
                                    max_len=hp.max_len,
                                    augment_index=hp.augment_index,
                                    augment_op=hp.augment_op)

    # train the model
    initialize_and_train(config,
                         train_dataset,
                         augment_dataset,
                         valid_dataset,
                         test_dataset,
                         u_dataset,
                         u_dataset_aug,
                         hp, run_tag)
