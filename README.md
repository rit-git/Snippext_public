# Snippext
Snippext is the extraction pipeline for mining opinions and customer experiences from user-generated content (e.g., online reviews).

Paper: Zhengjie Miao, Yuliang Li, Xiaolan Wang, Wang-Chiew Tan, "Snippext: Semi-supervised Opinion Mining with Augmented Data", In theWebConf (WWW) 2020

## Requirements

* Python 3.7.5
* PyTorch 1.3
* HuggingFace Transformers 
* Spacy with the ``em_core_web_sm`` models
* NLTK (stopwords, wordnet)
* Gensim
* NVIDIA Apex (fp16 training)

Install required packages
```
conda install -c conda-forge nvidia-apex
pip install -r requirements.txt
```

Download pre-trained BERT models and word2vec models (for data augmentation) :
```
wget https://snippext.s3.us-east-2.amazonaws.com/finetuned_bert.zip
unzip finetuned_bert.zip
wget https://snippext.s3.us-east-2.amazonaws.com/word2vec.zip
unzip word2vec.zip
```

## Training with the baseline BERT finetuning

The baseline method performs BERT finetuning on a specific task:
```
CUDA_VISIBLE_DEVICES=0 python train_baseline.py \
  --task restaurant_ae_tagging \
  --logdir results/ \
  --save_model \
  --finetuning \
  --batch_size 32 \
  --lr 5e-5 \
  --n_epochs 20 \
  --bert_path finetuned_bert/rest_model.bin
```

Parameters:
* ``--task``: the name of the task (defined in ``configs.json``)
* ``--logdir``: the logging directory with Tensorboard
* ``--save_model``: whether to save the best model
* ``--batch_size``, ``--lr``, ``--n_epochs``: batch size, learning rate, and the number of epochs
* ``--bert_path`` (Optional): the path of a fine-tuned BERT checkpoint. Use the base uncased model if not specified.
* ``--max_len`` (Optional): maximum sequence length

*(New)* (also in MixDA and MixMatchNL):
* ``--fp16`` (Optional): whether to train with fp16 acceleration
* ``--lm`` (Optional): other language models, e.g., "distilbert" or "albert"

### Task Specification

The train/dev/test sets of a task (tagging or span classification) are specificed in the file ``configs.json``. 
The file ``configs.json`` is a list of entries where each one is of the following format:
```
{
  "name": "hotel_tagging",
  "task_type": "tagging",
  "vocab": [
    "B-AS",
    "I-AS",
    "B-OP",
    "I-OP"
  ],
  "trainset": "combined_data/hotel/train.txt.combined",
  "validset": "combined_data/hotel/dev.txt",
  "testset": "combined_data/hotel/dev.txt",
  "unlabeled": "combined_data/hotel/unlabeled.txt"
},
```

Fields:
* ``name``: the name of the task. A tagging task should end with a suffix ``_tagging``
* ``task_type``: either ``tagging`` or ``classification``
* ``vocab``: the list of class labels. For tagging tasks, all labels start with ``B-`` or ``I-`` indicating the begin/end of a span. For classification task, the list contains all the possible class labels.
* ``trainset``, ``validset``, ``testset``: the paths to the train/dev/test sets
* ``unlabeled`` (Optional): the path to the unlabeled dataset for semi-supervised learning. The file has same format as the train/test sets but the labels are simply ignored.

## Training with MixDA (data augmentation)

1. Build the augmentation index:

```
python augment_index_builder.py \
  --task restaurant_ae_tagging \
  --w2v_path word2vec/rest_w2v.model \
  --idf_path word2vec/rest_finetune.txt \
  --bert_path finetuned_bert/rest_model.bin \
  --index_output_path augment/rest_index.json
```

Simply replace ``restaurant_ae_tagging`` with ``laptop_ae_tagging``, ``restaurant_asc``, and ``laptop_asc`` to generate the other indices.
Replace ``rest`` with ``laptop`` for the ``laptop_ae_tagging`` and ``laptop_asc`` indices.

2. Train with:
```
CUDA_VISIBLE_DEVICES=0 python train_mixda.py \
  --task restaurant_ae_tagging \
  --logdir results/ \
  --finetuning \
  --batch_size 32 \
  --lr 5e-5 \
  --n_epochs 5 \
  --bert_path finetuned_bert/rest_model.bin \
  --alpha_aug 0.8 \
  --augment_index augment/rest_index.json \
  --augment_op token_repl_tfidf \
  --run_id 0 
```

Parameters:
* ``alpha_aug``: the [mixup](https://arxiv.org/abs/1710.09412) parameter between the original example and the augmented example ({0.2, 0.5, 0.8} are usually good). 
* ``augment_index``: the path to the augmentation indices
* ``augment_op``: the name of the DA operators. We currently support the following 9 operators:

| Operators       | Details                                           |
|-----------------|---------------------------------------------------|
|token_del_tfidf  | Token deletion by importance (measured by TF-IDF) |
|token_del        | Token deletion (uniform)                          |
|token_repl_tfidf | Token replacement by importance                   |
|token_repl       | Token replacement (uniform)                       |
|token_swap       | Swapping two tokens                               |
|token_ins        | Inserting new tokens                              |
|span_sim         | Replacing a span with similar a one               |
|span_freq        | Replacing a span by frequency                     |
|span             | Uniform span replacement                          |


## Training with MixMatchNL (MixDA + Semi-supervised Learning)

Our implementation of [MixMatch](https://arxiv.org/abs/1905.02249) with MixDA. To train with MixMatchNL:
```
CUDA_VISIBLE_DEVICES=0 python train_mixmatchnl.py \
  --task restaurant_ae_tagging \
  --logdir results/ \
  --finetuning \
  --batch_size 32 \
  --lr 5e-5 \
  --n_epochs 5 \
  --bert_path finetuned_bert/rest_model.bin \
  --alpha_aug 0.8 \
  --alpha 0.2 \
  --u_lambda 50.0 \
  --num_aug 2 \
  --augment_index augment/rest_index.json \
  --augment_op token_repl_tfidf \
  --run_id 0 
```

Additional parameters:
* ``alpha``: the mixup parameter (between labeled and unlabeled data). We chose ``alpha`` from {0.2, 0.5, 0.8}.
* ``u_lambda``: the weight of unlabeled data loss, typically (chosen from {10.0, 25.0, 50.0})
* ``num_aug``: the number of augmented examples per unlabeled example (we chose from {2, 4})


### Hyperparameters and experiment scipts (coming soon)
