"""
Check the objects of faster-rcnn outputs
and match with parsed objects in questions.
Count the number of matched as the score for ranking
or reject invalid questions.
"""

import os
import collections

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from nltk.stem.porter import PorterStemmer
# from nltk import word_tokenize
import spacy

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator
from uniter.uniter import GQAUNITER
from butd.preprocess import gqa_create_dictionary_glove
from butd.butd import GQABUTD
import json

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    # for gqa-uq we add one more class
    dset.ans2label['UQ'] = dset.num_answers
    dset.label2ans.append('UQ')
    dset.num_answers = len(dset.label2ans)
    print(f"One more answer added for {splits}. Now there are {dset.num_answers} answers.")
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class GQA:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=not args.chart
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        # self.model = GQAModel(self.train_tuple.dataset.num_answers)
        # there is no need to add extra class
        if args.backbone == 'lxmert':
            self.model = GQAModel(self.train_tuple.dataset.num_answers - 1)
        elif args.backbone == "butd":
            dictionary, emb = gqa_create_dictionary_glove()
            self.model = GQABUTD(self.train_tuple.dataset.num_answers - 1, dictionary)
            self.model.w_emb.load_embeddings(emb)
        elif args.backbone == "uniter":
            self.model = GQAUNITER(self.train_tuple.dataset.num_answers - 1)
        else:
            raise ValueError(f"Backbone {args.backbone} not implemented!")

        # Load pre-trained weights
        if args.backbone == "lxmert":
            if args.load_lxmert is not None:
                self.model.lxrt_encoder.load(args.load_lxmert)
            if args.load_lxmert_qa is not None:
                load_lxmert_qa(args.load_lxmert_qa, self.model,
                            label2ans=self.train_tuple.dataset.label2ans[:-1])
        elif args.backbone == "uniter":
            if args.load_lxmert is not None:
                self.model.encoder.load(args.load_lxmert)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        self.ps  = PorterStemmer()
        self.nlp = spacy.load('en_core_web_sm')
        self.gqa_objects = []
        with open("data/vg_gqa_imgfeat/objects_vocab.txt", 'r') as f:
            for line in f.readlines():
                self.gqa_objects.append(line.strip().split(",")[0])
        self.gqa_objects = [self.ps.stem(obj) for obj in self.gqa_objects]
        
    def ood_evaluate(self, eval_tuple: DataTuple, dump=None):
        from gqa_data import GQAOODEvaluator
        self.model.eval()
        dset, loader, _ = eval_tuple
        evaluator = GQAOODEvaluator(dset)
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                _, label = torch.sigmoid(logit).max(1)

                score = []
                for qid, s in zip(ques_id, sent):
                    datum = loader.dataset.raw_dataset.id2datum[qid] # get data according to question id
                    objects_id = loader.dataset.imgid2img[datum['img_id']]['objects_id'] # get objects according to image id
                    objects = [self.gqa_objects[oid] for oid in objects_id] # convert to object name
                    tokens = [self.ps.stem(tok.text) for tok in self.nlp(s) \
                        if tok.pos_ == 'NOUN' and tok.text not in \
                            ['left', 'right', 'thing', 'top', 'bottom', 'photo', 'image', 'kind', 'color']] # find stem of questions
                    score.append(1. if all([tok in objects for tok in tokens]) else 0.) # check object is detected
                    # if i > 20:

                for qid, l, s in zip(ques_id, label.cpu().numpy(), np.asarray(score)):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = (ans, s)
        results = evaluator.evaluate(quesid2ans)
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return results

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    # Build Class
    gqa = GQA()

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    # Test or Train
    if args.test is not None:
        # args.fast = args.tiny = False       # Always loading all data in test
        # if 'submit' in args.test:
        #     gqa.predict(
        #         get_tuple(args.test, bs=args.batch_size,
        #                   shuffle=False, drop_last=False),
        #         dump=os.path.join(args.output, 'submit_predict.json')
        #     )
        # if 'testdev' in args.test:
        #     result = gqa.evaluate(
        #         get_tuple('testdev', bs=args.batch_size,
        #                   shuffle=False, drop_last=False),
        #         dump=os.path.join(args.output, 'testdev_predict.json')
        #     )
        #     print(result)
        if args.test.startswith("GQAUQ"):
            result = gqa.ood_evaluate(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, f'{args.test}_predict.json')
            )
            print(result)

            with open(os.path.join(args.output, f'{args.test}_result.json'), "w") as outfile:
                json.dump(result, outfile)
 
        else:
            raise RuntimeError("Can not be applied on datasets w/o unanswerable questions!")
    else:
        raise RuntimeError("Can not be trained!")
