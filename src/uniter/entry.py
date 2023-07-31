import os

import torch
import torch.nn as nn

from uniter.tokenization import BertTokenizer
from uniter.modeling import VISUAL_CONFIG
from uniter.modeling import UniterFeatureExtraction as UFE

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = 9
    VISUAL_CONFIG.x_layers = 5
    VISUAL_CONFIG.r_layers = 5


class UniterEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_seq_length = 20
        set_visual_config(args)
        self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-cased",
                do_lower_case=True
            )
        self.model = UFE.from_pretrained(
                "bert-base-cased")
        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)


    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, boxes, visual_attention_mask=None):
        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()
        
        assert feats.shape[1] == 36 , "Not Using 36 ROIs, please change the following 2 lines"
        visual_segment_ids = torch.ones(input_ids.shape[0],feats.shape[1],dtype=torch.long).cuda()
        v_mask = torch.ones(input_mask.shape[0],feats.shape[1],dtype=torch.long).cuda()
        

        output = self.model(input_ids = input_ids, token_type_ids = segment_ids,attention_mask = input_mask,
                            visual_feats = feats,visual_token_type_ids=visual_segment_ids,
                            visual_attention_mask=v_mask,img_pos_feat=boxes)
        return output
    
    def load(self,path):
        state_dict = torch.load(path)
        for key in list(state_dict.keys()):
            if 'bert.' in key:
                state_dict[key.replace('bert.', 'uniter.')] = state_dict.pop(key)
        
        print("Load UNITER PreTrained Model from %s"%path)
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()
        self.model.load_state_dict(state_dict, strict=False)