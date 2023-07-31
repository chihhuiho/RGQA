# coding=utf-8
# Copyleft 2019 project LXRT.
# copied from LXRT with modifications
import torch
import torch.nn as nn

from param import args
from transformers import BertTokenizer, BertModel
from uniter.modeling import GeLU, BertLayerNorm 

MAX_VQA_LENGTH = 40


class GQABERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        hid_dim = 768

        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 1)
        )
        self.logit_fc.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, cap, sent):
        pairs = [[c, s] for c, s in zip(cap, sent)]
        encoded = self.tokenizer(
            pairs,
            max_length=MAX_VQA_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded.input_ids.cuda()
        input_mask = encoded.attention_mask.cuda()
        segment_ids = encoded.token_type_ids.cuda()

        x = self.encoder(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids
        ).last_hidden_state[:, 0, :]
        logit = self.logit_fc(x)

        return logit