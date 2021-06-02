### Setup
Before trying to run this locally, you have to create a virtual environment
and to install the libraries specified in requirements.txt.

In addition to that you have to download the pre-trained BERT model from here https://huggingface.co/readerbench/RoBERT-base.
All downloaded files should be placed in model/ folder.

Also, you have to download fasttext embedding for romanian language.
https://fasttext.cc/docs/en/crawl-vectors.html

### Tasks definitions
The tagging task is called: laptop_ae_tagging.
The classification task is called: sentiment_analysis.

The configuration can be found in configs.json.
The datasets used for training these tasks need to be placed in
data/sentiment_analysis and data/laptop_ae_tagging.

### How to run it locally
Before training, you need to generate the augment index used
for data augmentation.

The tf-idf dictionary is saved with name idf.json for laptop_ae_tagging
and idf_sa.json for sentiment_analysis.

The models can be trained using train_baseline, train_mixda
or train_mixmatchnl.

To test the models, you can run:
```
python run_pipeline.py \
  --task <task_name> \
  --checkpoint_path <checkpoint_path> \
```
where checkpoint_path is ./checkpoints for tagging
and ./checkpoints_sentiment_analysis/token_ins_mixmatchnl for classification
and task name is either laptop_ae_tagging or sentiment_analysis.

This is a forked repository after https://github.com/rit-git/Snippext_public
which was adapted for romanian language.