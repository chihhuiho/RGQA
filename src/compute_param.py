"""
The file for training gqa-uq. Mostly copied from gqa.py.
"""

import os
import json
import pickle
import collections
import random

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import numpy as np

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator, GQAViLTDataset, collate_fn_vilt
# from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator

from transformers import BertTokenizer
from uniter.uniter import GQAUNITER
from butd.preprocess import gqa_create_dictionary_glove
from butd.butd import GQABUTD
from vilt.vilt import GQAViLT
from vilt.modules.vilt_utils import set_schedule
import json

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


if __name__ == "__main__":
    # Build Class

    lxmert_model = GQAModel(1873)
    lxmert_model_parameters = filter(lambda p: p.requires_grad, lxmert_model.parameters())
    lxmert_params = sum([np.prod(p.size()) for p in lxmert_model_parameters])
    print("lxmert_params : " + str(lxmert_params))

    
    dictionary, emb = gqa_create_dictionary_glove()
    butd_model = GQABUTD(1873, dictionary)
    butd_model_parameters = filter(lambda p: p.requires_grad, butd_model.parameters())
    butd_params = sum([np.prod(p.size()) for p in butd_model_parameters])
    print("butd_params : " + str(butd_params))



    uniter_model = GQAUNITER(1873)
    uniter_model_parameters = filter(lambda p: p.requires_grad, uniter_model.parameters())
    uniter_params = sum([np.prod(p.size()) for p in uniter_model_parameters])
    print("uniter_params : " + str(uniter_params))



    vilt_model = GQAViLT(1873)
    vilt_model_parameters = filter(lambda p: p.requires_grad, vilt_model.parameters())
    vilt_params = sum([np.prod(p.size()) for p in vilt_model_parameters])
    print("vilt_params : " + str(vilt_params))

   
