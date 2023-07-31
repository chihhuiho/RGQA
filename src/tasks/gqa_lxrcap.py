# use pretrained LXMERT score to reject unanswerable questions

import os
import collections

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from PIL import Image

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel, MAX_GQA_LENGTH
from lxrt.modeling import LXRTPretraining
from lxrt.tokenization import BertTokenizer
from lxrt.entry import convert_sents_to_features
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator


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
        self.model = GQAModel(self.train_tuple.dataset.num_answers - 1)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans[:-1])

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        self.cap_model = LXRTPretraining.from_pretrained(
            "bert-base-uncased",
            task_mask_lm=True,
            task_obj_predict=True,
            task_matched=True,
            task_qa=True,
            visual_losses="obj,attr,feat",
            num_answers=self.train_tuple.dataset.num_answers - 1
        )
        pretrained_state = torch.load("snap/pretrained/model_LXRT.pth")
        for key in list(pretrained_state.keys()):
            if '.module' in key:
                pretrained_state[key.replace('.module', '')] = pretrained_state.pop(key)
        self.cap_model.load_state_dict(pretrained_state, strict=False)
        self.cap_model.cuda()
        self.cap_model.eval()
        self.cap_proc = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        
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

                sent_features = convert_sents_to_features(sent, MAX_GQA_LENGTH, self.cap_proc)
                input_ids = torch.tensor([f.input_ids for f in sent_features], dtype=torch.long).cuda()
                input_mask = torch.tensor([f.input_mask for f in sent_features], dtype=torch.long).cuda()
                segment_ids = torch.tensor([f.segment_ids for f in sent_features], dtype=torch.long).cuda()

                score = self.cap_model.forward_match(input_ids, segment_ids, input_mask, feats, boxes)
                breakpoint()
                score = torch.softmax(score.view(-1, 2), dim=-1)[:, -1]

                for qid, l, s in zip(ques_id, label.cpu().numpy(), score.cpu().numpy()):
                    ans = dset.label2ans[l] if s > 0.5 else 'UQ'
                    quesid2ans[qid] = (ans, s)
        
        results = evaluator.evaluate(quesid2ans, minmax=True)
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
        else:
            raise RuntimeError("Can not be applied on datasets w/o unanswerable questions!")
    else:
        raise RuntimeError("Can not be trained!")