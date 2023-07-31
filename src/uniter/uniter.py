# coding=utf-8
# Copyleft 2019 project LXRT.
# copied from LXRT with modifications
import torch
import torch.nn as nn

from param import args
from uniter.entry import UniterEncoder
from uniter.modeling import GeLU, BertLayerNorm

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class GQAUNITER(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        # Build LXRT encoder
        self.encoder = UniterEncoder(args)
        
        hid_dim = self.encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size
        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.encoder(sent, feat, pos)
        logit = self.logit_fc(x)

        return logit

class GQAUNITER_maha(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        # Build LXRT encoder
        self.encoder = UniterEncoder(args)
        
        hid_dim = self.encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size
        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.encoder(sent, feat, pos)
        logit = self.logit_fc(x)

        return logit, x
