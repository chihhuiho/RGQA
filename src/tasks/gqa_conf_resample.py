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

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False, vilt=False) -> DataTuple:
    dset = GQADataset(splits)
    # for gqa-uq we add one more class
    dset.ans2label['UQ'] = dset.num_answers
    dset.label2ans.append('UQ')
    dset.num_answers = len(dset.label2ans)
    print(f"One more answer added for {splits}. Now there are {dset.num_answers} answers.")
    if vilt:
        tset = GQAViLTDataset(dset)
        data_loader = DataLoader(
            tset, batch_size=bs,
            shuffle=shuffle, num_workers=args.num_workers,
            drop_last=drop_last, pin_memory=True, collate_fn=collate_fn_vilt
        )
    else:
        tset = GQATorchDataset(dset)
        data_loader = DataLoader(
            tset, batch_size=bs,
            shuffle=shuffle, num_workers=args.num_workers,
            drop_last=drop_last, pin_memory=True
        )
    evaluator = GQAEvaluator(dset)
    

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class GQA:
    def __init__(self):
        self.train_pos_tuple = get_tuple(
            args.train_pos, bs=args.batch_size, shuffle=True, drop_last=not args.chart,
            vilt=args.backbone == 'vilt'
        )
        self.train_neg_tuple = get_tuple(
            args.train_neg, bs=args.batch_size, shuffle=True, drop_last=not args.chart,
            vilt=args.backbone == 'vilt'
        )
        if args.valid != "":
            valid_bsize = 512 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False,
                vilt=args.backbone == 'vilt'
            )
        else:
            self.valid_tuple = None

        # self.model = GQAModel(self.train_tuple.dataset.num_answers)
        # there is no need to add extra class
        if args.backbone == "lxmert":
            self.model = GQAModel(self.train_pos_tuple.dataset.num_answers - 1)
        elif args.backbone == "butd":
            dictionary, emb = gqa_create_dictionary_glove()
            self.model = GQABUTD(self.train_pos_tuple.dataset.num_answers - 1, dictionary)
            self.model.w_emb.load_embeddings(emb)
        elif args.backbone == "uniter":
            self.model = GQAUNITER(self.train_pos_tuple.dataset.num_answers - 1)
        elif args.backbone == "vilt":
            self.model = GQAViLT(self.train_pos_tuple.dataset.num_answers - 1)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        else:
            raise ValueError(f"Backbone {args.backbone} not implemented!")

        # Load pre-trained weights
        if args.backbone == "lxmert":
            if args.load_lxmert is not None:
                self.model.lxrt_encoder.load(args.load_lxmert)
            if args.load_lxmert_qa is not None:
                load_lxmert_qa(args.load_lxmert_qa, self.model,
                            label2ans=self.train_pos_tuple.dataset.label2ans[:-1])
        elif args.backbone == "vilt":
            if args.load_lxmert is not None:
                ckpt = torch.load(args.load, map_location='cpu')
                state_dict = ckpt['state_dict']
                self.model.load_state_dict(state_dict, strict=False)
        elif args.backbone == "uniter":
            if args.load_lxmert is not None:
                self.model.encoder.load(args.load_lxmert)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if args.backbone in ['vilt']:
            self.optim, self.sched = set_schedule(self.model, args, len(self.train_tuple[1]))
        elif args.backbone in ['lxmert', 'uniter', 'butd']:
            if 'bert' in args.optim:
                batch_per_epoch = len(self.train_pos_tuple.loader)
                t_total = int(batch_per_epoch * args.epochs)
                print("Total Iters: %d" % t_total)
                from lxrt.optimization import BertAdam
                self.optim = BertAdam(list(self.model.parameters()),
                                    lr=args.lr,
                                    warmup=0.1,
                                    t_total=t_total)
            else:
                self.optim = args.optimizer(list(self.model.parameters()), args.lr, weight_decay=0.)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        if args.chart:
            os.makedirs(os.path.join(args.output, 'chart'), exist_ok=True)
        if args.save_all:
            print("Model after each epoch will be saved!")

    def train(self, train_pos_tuple, train_neg_tuple, eval_tuple):
        # dset, loader, evaluator = train_tuple
        dset_pos, loader_pos, _ = train_pos_tuple
        dset_neg, loader_neg, _ = train_neg_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader_pos))) if args.tqdm else (lambda x: x)
        neg_iter = iter(loader_neg)

        best_valid = 0.
        for epoch in range(args.epochs):
            # quesid2ans = {}
            quesid2score = {}
            for i, (ques_id_pos, feats_pos, boxes_pos, sent_pos, target_pos) in iter_wrapper(enumerate(loader_pos)):

                try:
                    ques_id_neg, feats_neg, boxes_neg, sent_neg, target_neg = neg_iter.next()
                except StopIteration:
                    neg_iter = iter(loader_neg)
                    ques_id_neg, feats_neg, boxes_neg, sent_neg, target_neg = neg_iter.next()
                
                ques_id = ques_id_pos + ques_id_neg
                feats = torch.cat([feats_pos, feats_neg], dim=0)
                boxes = torch.cat([boxes_pos, boxes_neg], dim=0)
                sent = sent_pos + sent_neg
                target = torch.cat([target_pos, target_neg], dim=0)

                self.model.train()
                self.optim.zero_grad()

                target = target[:, :-1]

                if args.backbone in ['lxmert', 'butd', 'uniter']:
                    feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                    logit = self.model(feats, boxes, sent)
                elif args.backbone in ['vilt']:
                    feats, target = feats.cuda(), target.cuda()
                    encoded = self.tokenizer(
                        sent,
                        padding="max_length",
                        truncation=True,
                        max_length=20,
                        return_special_tokens_mask=True,
                        return_tensors='pt'
                    )
                    batch = {
                        'text_ids': encoded['input_ids'].to('cuda'),
                        'text_labels': encoded['input_ids'].to('cuda'),
                        'text_masks': encoded['attention_mask'].to('cuda'),
                        'image': [feats]
                    }
                    logit = self.model(batch)
                assert logit.dim() == target.dim() == 2
                if args.mce_loss:
                    max_value, target = target.max(1)
                    loss = self.mce_loss(logit, target) * logit.size(1)
                else:
                    loss = self.bce_loss(logit, target)
                    loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

            if args.backbone in ['vilt']:
                self.sched.step()

            # log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)
            log_str = "\nEpoch %d: Train Loss %0.2f\n" % (epoch, loss.item())

            # if self.valid_tuple is not None:  # Do Validation
            #     valid_score = self.evaluate(eval_tuple)
            #     if valid_score > best_valid:
            #         best_valid = valid_score
            #         self.save("BEST")
            
            #     log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
            #                "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
            
            if args.save_all:
                self.save(f"EPOCH_{epoch}")

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()
            
            if args.chart:
                with open(os.path.join(args.output, 'chart', f'epoch_{epoch}.pkl'), 'wb') as f:
                    pickle.dump(quesid2score, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.save("LAST")
    
    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = torch.sigmoid(logit).max(1)
                for qid, l, s in zip(ques_id, label.cpu().numpy(), score.cpu().numpy()):
                    ans = dset.label2ans[l] if s > args.tau else 'UQ'
                    quesid2ans[qid] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans
    
    def predict_with_thresh(self, eval_tuple: DataTuple, dump=None):
        from accfpr.accfpr import accfpr_curve
        self.model.eval()
        dset, loader, _ = eval_tuple
        quesid2ans = {}
        scores = []
        preds = []
        targets = []
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent, target = datum_tuple[:5]   # avoid handling target
            with torch.no_grad():
                target = np.argmax(target, axis=-1)
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = torch.sigmoid(logit).max(1)
                for qid, l, s, t in zip(ques_id, label.cpu().numpy(), score.cpu().numpy(), target):
                    pred = dset.label2ans[l]
                    ans = dset.label2ans[t]
                    quesid2ans[qid] = (pred, float(s), ans)
                    preds.append(pred)
                    scores.append(s)
                    targets.append(ans)
        targets = np.asarray(targets, dtype=str)
        scores = np.asarray(scores, dtype=float)
        preds = np.asarray(preds, dtype=str)
        _, acc, thresh = accfpr_curve(targets, preds, scores, 'UQ')
        assert args.target_acc is not None and args.target_acc < acc[-1]
        tau = np.interp(args.target_acc, acc, thresh)
        for qid in quesid2ans:
            if quesid2ans[qid][1] < tau:
                quesid2ans[qid] = ('UQ', quesid2ans[qid][1], quesid2ans[qid][2])
        if dump is not None:
            with open(dump, 'w') as f:
                json.dump(quesid2ans, f)
    
    def ood_evaluate(self, eval_tuple: DataTuple, dump=None):
        from gqa_data import GQAOODEvaluator
        self.model.eval()
        dset, loader, _ = eval_tuple
        evaluator = GQAOODEvaluator(dset, args.tau)
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                if args.backbone in ['lxmert', 'butd', 'uniter']:
                    feats, boxes = feats.cuda(), boxes.cuda()
                    logit = self.model(feats, boxes, sent)
                elif args.backbone in ['vilt']:
                    feats = feats.cuda()
                    encoded = self.tokenizer(
                        sent,
                        padding="max_length",
                        truncation=True,
                        max_length=20,
                        return_special_tokens_mask=True,
                        return_tensors='pt'
                    )
                    batch = {
                        'text_ids': encoded['input_ids'].to('cuda'),
                        'text_labels': encoded['input_ids'].to('cuda'),
                        'text_masks': encoded['attention_mask'].to('cuda'),
                        'image': [feats]
                    }
                    logit = self.model(batch)
                score, label = torch.sigmoid(logit).max(1)
                for qid, l, s in zip(ques_id, label.cpu().numpy(), score.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = (ans, s)
        results = evaluator.evaluate(quesid2ans)
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return results

    def get_pseudo_labels(self, train_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, _ = train_tuple

        for i, datum_tuple in tqdm(enumerate(loader), total=len(loader)):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = torch.sigmoid(logit).max(1)
                for qid, l, s in zip(ques_id, label.cpu().numpy(), score.cpu().numpy()):
                    ans = dset.label2ans[l]
                    dset.id2datum[qid]['label'] = {ans: float(s)}
        
        raw_data = [datum for _, datum in dset.id2datum.items()]
        if dump is not None:
            with open(dump, "w") as f:
                json.dump(raw_data, f)
        print(f"{len(raw_data)} pseudo data have been saved in {dump}.")

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

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

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

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
    # if args.load is not None and args.backbone in ['lxmert']:
    if args.load is not None:
        gqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            gqa.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False,
                          vilt=args.backbone == 'vilt'),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'testdev' in args.test:
            if 'GQAUQ' in args.test:
                if not args.predict:
                    result = gqa.ood_evaluate(
                        get_tuple(args.test, bs=args.batch_size,
                                shuffle=False, drop_last=False,
                                vilt=args.backbone == 'vilt'),
                        dump=os.path.join(args.output, f'{args.test}_predict.json')
                    )
                    print(result)
                else:
                    gqa.predict_with_thresh(
                        get_tuple(args.test, bs=args.batch_size,
                                shuffle=False, drop_last=False,
                                vilt=args.backbone == 'vilt'),
                        dump=os.path.join(args.output, f'{args.test}_thresholded.json')
                    )
            else:
                result = gqa.evaluate(
                    get_tuple('testdev', bs=args.batch_size,
                            shuffle=False, drop_last=False,
                            vilt=args.backbone == 'vilt'),
                    dump=os.path.join(args.output, 'testdev_predict.json')
                )
                print(result)
    elif args.pseudo is not None:
        gqa.get_pseudo_labels(
            get_tuple(args.pseudo, bs=args.batch_size,
                      shuffle=False, drop_last=False),
            dump=os.path.join(args.output, 'pseudo_data.json')
        )
    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train Positive data:', gqa.train_pos_tuple.dataset.splits)
        print('Splits in Train Negative data:', gqa.train_neg_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            # print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_pos_tuple, gqa.train_neg_tuple, gqa.valid_tuple)


