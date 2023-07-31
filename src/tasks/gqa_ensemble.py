"""
Load a list of models and ensemble their answers
"""

import os
import json
import pickle
import collections

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator

from butd.preprocess import gqa_create_dictionary_glove
from butd.butd import GQABUTD
from uniter.uniter import GQAUNITER

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
        if args.backbone == "lxmert":
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

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        if args.chart:
            os.makedirs(os.path.join(args.output, 'chart'), exist_ok=True)
        if args.save_all:
            print("Model after each epoch will be saved!")
        
    def ood_evaluate(self, eval_tuple: DataTuple, dump=None):
        # from gqa_data import GQAOODEvaluator
        self.model.eval()
        dset, loader, _ = eval_tuple
        # evaluator = GQAOODEvaluator(dset, args.tau)
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                scores = torch.sigmoid(logit)
                # _, label = scores.max(1)
                for qid, s in zip(ques_id, scores.cpu().numpy()):
                    # ans = dset.label2ans[l]
                    quesid2ans[qid] = s
        # results = evaluator.evaluate(quesid2ans)
        # if dump is not None:
        #     evaluator.dump_result(quesid2ans, dump)
        return quesid2ans
    
    def get_target(self, eval_tuple: DataTuple):
        dset, loader, _ = eval_tuple
        quesid2targets = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, _, _, _, target = datum_tuple[:5]   # avoid handling target
            target = np.argmax(target, axis=-1)
            for qid, t in zip(ques_id, target):
                quesid2targets[qid] = dset.label2ans[t]
        return quesid2targets

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    args.model_list = args.load.strip().split(",")

    # Build Class
    gqa = GQA()

    # Load data
    assert args.test is not None, "Currently only for test!"
    args.fast = args.tiny = False       # Always loading all data in test
    assert 'testdev' in args.test or 'submit', "Currently only testdev or test!"
    # assert 'GQAUQ' in args.test, "Currently only GQAUQ!"
    data_tuple = get_tuple(args.test, bs=args.batch_size,
                shuffle=False, drop_last=False)

    # Load Model
    results_list = []
    for model_path in args.model_list:
        gqa.load(model_path)

        # Test
        result = gqa.ood_evaluate(data_tuple)
        results_list.append(result)

    final_results = results_list[0]
    for results in results_list[1:]:
        for qid in results:
            if args.ensemble_method == "mean":
                final_results[qid] += results[qid]
            elif args.ensemble_method == "multiply":
                final_results[qid] *= results[qid]
            else:
                raise ValueError(f"Ensemble method {args.ensemble_method} not implemented!")

    dset, _, closesetEvaluator = data_tuple
    for qid in final_results:
        if args.ensemble_method == "mean":
            scores = final_results[qid] / len(results_list)
        else:
            scores = final_results[qid]
        label = np.argmax(scores)
        score = scores[label]
        final_results[qid] = (dset.label2ans[label], score)
    
    if args.predict:
        from accfpr.accfpr import accfpr_curve
        quesid2targets = gqa.get_target(data_tuple)
        score_list = []
        target_list = []
        pred_list = []
        for qid in quesid2targets:
            pred_list.append(final_results[qid][0])
            score_list.append(final_results[qid][1])
            target_list.append(quesid2targets[qid])
        target_list = np.asarray(target_list, dtype=str)
        score_list = np.asarray(score_list, dtype=float)
        pred_list = np.asarray(pred_list, dtype=str)
        _, acc, thresh = accfpr_curve(target_list, pred_list, score_list, 'UQ')
        assert args.target_acc is not None and args.target_acc < acc[-1]
        tau = np.interp(args.target_acc, acc, thresh)
        for qid in final_results:
            if final_results[qid][1] < tau:
                final_results[qid] = ('UQ', float(final_results[qid][1]), quesid2targets[qid])
            else:
                final_results[qid] = (final_results[qid][0], float(final_results[qid][1]), quesid2targets[qid])
        dump = os.path.join(args.output, args.test+"_thresholded.json")
        with open(dump, 'w') as f:
            json.dump(final_results, f)
    elif not args.predict and args.predict_closeset:
        quesid2pred = {}
        for qid in final_results:
            quesid2pred[qid] = final_results[qid][0]
        closesetEvaluator.dump_result(quesid2pred, os.path.join(args.output, 'submit_predict.json'))
    else:
        from gqa_data import GQAOODEvaluator
        evaluator = GQAOODEvaluator(dset, args.tau)
        performance = evaluator.evaluate(final_results)
        print(performance)
        with open(os.path.join(args.output, f'{args.test}_result.json'), "w") as outfile:
            json.dump(performance, outfile)
 

        evaluator.dump_result(final_results, os.path.join(args.output, args.test+"_predict.json"))
