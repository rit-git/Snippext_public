import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AlbertModel, DistilBertModel, RobertaModel, XLNetModel

model_ckpts = {'bert': "bert-base-uncased",
               'albert': "albert-base-v2",
               'roberta': "roberta-base",
               'xlnet': "xlnet-base-cased",
               'distilbert': "distilbert-base-uncased"}

def get_lm(lm='bert', bert_path=None):
    # load the model or model checkpoint
    if bert_path is None:
        model_state_dict = None
    else:
        output_model_file = bert_path
        model_state_dict = torch.load(output_model_file,
                        map_location=lambda storage, loc: storage)
    if lm == 'bert':
        bert = BertModel.from_pretrained(model_ckpts[lm],
                state_dict=model_state_dict)
    elif lm == 'distilbert':
        bert = DistilBertModel.from_pretrained(model_ckpts[lm],
                state_dict=model_state_dict)
    elif lm == 'albert':
        bert = AlbertModel.from_pretrained(model_ckpts[lm],
                state_dict=model_state_dict)
    elif lm == 'xlnet':
        bert = XLNetModel.from_pretrained(model_ckpts[lm],
                state_dict=model_state_dict)
    elif lm == 'roberta':
        bert = RobertaModel.from_pretrained(model_ckpts[lm],
                state_dict=model_state_dict)
    return bert


class MultiTaskNet(nn.Module):
    def __init__(self, task_configs=[],
                 device='cpu',
                 lm='bert',
                 bert_path=None):
        super().__init__()

        assert len(task_configs) > 0

        self.bert = get_lm(lm, bert_path)
        self.device = device
        self.task_configs = task_configs
        self.module_dict = nn.ModuleDict({})

        # hard corded for now
        hidden_size = 768
        hidden_dropout_prob = 0.1

        config = task_configs[0]
        name = config['name']
        task_type = config['task_type']
        vocab = config['vocab']

        if task_type == 'tagging':
            # for tagging
            vocab_size = len(vocab) # 'O' and '<PAD>'
            if 'O' not in vocab:
                vocab_size += 1
            if '<PAD>' not in vocab:
                vocab_size += 1
        else:
            # for pairing and classification
            vocab_size = len(vocab)

        self.num_classes = vocab_size
        self.module_dict['%s_dropout' % name] = nn.Dropout(hidden_dropout_prob)
        self.module_dict['%s_fc' % name] = nn.Linear(hidden_size, vocab_size)


    def forward(self, x=None,
                y=None,
                x_enc=None,
                x_emb=None,
                get_enc=False,
                get_emb=False,
                token_type_ids=None,
                attention_mask=None,
                task='hotel_tagging'):
        """Forward function of the BERT models for classification/tagging.

        Args:
            x (LongTensor, optional): the input ids
            y (Tensor, optional): the label tensor
            x_enc (Tensor, optional): if not None, use as the LM output
            x_emb (Tensor, optional): if not None, use as the embedding layer's output
            get_enc (boolean, optional): if True, returns the output of the LM
            get_emb (boolean, optional): if True, returns the output of the embedding layer
            task (string, optional): the task name

        Returns:
            Tensor: logits, LM encoding, or embedding
            Tensor: y
            Tensor: yhat
        """

        # move labels to GPU
        if y is not None:
            y = y.to(self.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        dropout = self.module_dict[task + '_dropout']
        fc = self.module_dict[task + '_fc']

        if x is not None:
            x = x.to(self.device)
            # x_emb = self.bert.get_input_embeddings()(x)
            x_emb = self.bert.embeddings(x)
            if get_emb:
                return x_emb
            x_enc = self.bert(x,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)[0]
            # x_enc = self.bert(x)[0]

            # classification and regression
            if 'tagging' not in task:
                x_enc = x_enc[:, 0, :]
        elif x_emb is not None:
            x_enc = self.bert(inputs_embeds=x_emb)[0]
            # classification and regression
            if 'tagging' not in task:
                x_enc = x_enc[:, 0, :]

        if get_enc:
            return x_enc

        # dropout and the linear layer
        x_enc = dropout(x_enc)
        logits = fc(x_enc)
        if 'sts-b' in task:
            y_hat = logits
        else:
            y_hat = logits.argmax(-1)
        return logits, y, y_hat

