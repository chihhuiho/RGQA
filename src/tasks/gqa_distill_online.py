import os
import json
import pickle
import collections
import random

import torch
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

        # student
        if args.backbone == "lxmert":
            self.model = GQAModel(self.train_tuple.dataset.num_answers - 1)
        elif args.backbone == "butd":
            dictionary, emb = gqa_create_dictionary_glove()
            self.model = GQABUTD(self.train_tuple.dataset.num_answers - 1, dictionary)
            self.model.w_emb.load_embeddings(emb)
        else:
            raise ValueError(f"Backbone {args.backbone} not implemented!")
        
        # teacher
        if args.backbone == "lxmert":
            self.teacher = GQAModel(self.train_tuple.dataset.num_answers - 1)
        elif args.backbone == "butd":
            dictionary, emb = gqa_create_dictionary_glove()
            self.teacher = GQABUTD(self.train_tuple.dataset.num_answers - 1, dictionary)
            self.teacher.w_emb.load_embeddings(emb)
        else:
            raise ValueError(f"Backbone {args.backbone} not implemented!")
        
        self.load_teacher(args.teacher_path)
        
        # Load pre-trained weights
        if args.backbone == "lxmert":
            if args.load_lxmert is not None:
                self.model.lxrt_encoder.load(args.load_lxmert)
            if args.load_lxmert_qa is not None:
                load_lxmert_qa(args.load_lxmert_qa, self.model,
                            label2ans=self.train_tuple.dataset.label2ans[:-1])
        
        # GPU options
        self.model = self.model.cuda()
        self.teacher = self.teacher.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
        
        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
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
    
    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            # quesid2ans = {}
            quesid2score = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                target = target[:, :-1]

                if args.mixup_mode == 'perturb':
                    # 1. perturb boxes
                    # boxes: (B, N, 4)
                    pet_boxes = boxes[:, torch.randperm(boxes.shape[1]), :]

                    # 2. expand inputs
                    boxes = torch.cat([boxes, pet_boxes], 0)
                    feats = feats.repeat(2, 1, 1)
                    sent = sent + sent
                    target = torch.cat([target, torch.zeros_like(target)], 0)
                elif args.mixup_mode.startswith('mixup'):
                    # 1. sample negative image
                    pet_feats, pet_boxes, pet_target = [], [], []
                    num_roi = feats.shape[1]
                    for j, qid in enumerate(ques_id):
                        # the question should have different image id
                        img_id = dset.id2datum[qid]['img_id']
                        qid_r = random.choice(ques_id)
                        while dset.id2datum[qid_r]['img_id'] == img_id:
                            qid_r = random.choice(ques_id)
                        feat_pos = feats[j]
                        feat_neg = feats[ques_id.index(qid_r)]
                        # feats: (B, N, size)
                        prop = random.random()
                        pet_feat = torch.cat([feat_pos[:int(prop * num_roi), :],
                                              feat_neg[int(prop * num_roi):, :]], 0)
                        box_pos = boxes[j]
                        box_neg = boxes[ques_id.index(qid_r)]
                        pet_box = torch.cat([box_pos[:int(prop * num_roi), :],
                                             box_neg[int(prop * num_roi):, :]], 0)
                        pet_feats.append(pet_feat)
                        pet_boxes.append(pet_box)
                        if args.mixup_mode == 'mixup_v1':
                            pet_target.append(target[j] * prop)
                        elif args.mixup_mode == 'mixup_v2':
                            pet_target.append(target[j] * 0)
                        else:
                            raise ValueError(args.mixup_mode)

                    # 2. expand inputs
                    feats = torch.cat([feats, torch.stack(pet_feats, 0)], 0)
                    boxes = torch.cat([boxes, torch.stack(pet_boxes, 0)], 0)
                    target = torch.cat([target, torch.stack(pet_target, 0)], 0)
                    sent = sent + sent
                
                elif args.mixup_mode.startswith('weighted_sum'):
                    # 1. sample negative image
                    pet_feats = []
                    num_roi = feats.shape[1]
                    for j, qid in enumerate(ques_id):
                        # the question should have different image id
                        img_id = dset.id2datum[qid]['img_id']
                        qid_r = random.choice(ques_id)
                        while dset.id2datum[qid_r]['img_id'] == img_id:
                            qid_r = random.choice(ques_id)
                        feat_pos = feats[j]
                        feat_neg = feats[ques_id.index(qid_r)]
                        # feats: (B, N, size)
                        prop = random.random()
                        pet_feat = feat_pos * prop + feat_neg * (1 - prop)
                        pet_feats.append(pet_feat)
                        if args.mixup_mode == 'weighted_sum_v1':
                            pet_target.append(target[j] * prop)
                        elif args.mixup_mode == 'weighted_sum_v2':
                            pet_target.append(target[j] * 0)
                        else:
                            raise ValueError(args.mixup_mode)
                    
                    # 2. expand inputs
                    feats = torch.cat([feats, torch.stack(pet_feats, 0)], 0)
                    boxes = boxes.repeat(2, 1, 1)
                    target = torch.cat([target, torch.stack(pet_target, 0)], 0)
                    sent = sent + sent
                elif args.mixup_mode == 'none':
                    pass

                # 3. forward and compute loss
                outdom_id = target.sum(-1) < 1.
                feats, boxes = feats.cuda(), boxes.cuda()
                with torch.no_grad():
                    logit_t = self.teacher(feats[outdom_id], boxes[outdom_id], [sent[oid] for oid in torch.where(outdom_id)[0]])
                    assert logit_t.dim() == 2
                    soft_target = torch.sigmoid(logit_t)
                    target[outdom_id] = soft_target * args.lam + target[outdom_id] * (1 - args.lam)
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                if args.chart:
                    score, label = torch.sigmoid(logit).max(1)
                    for qid, l, s in zip(ques_id, label.cpu().numpy(), score.cpu().detach().numpy()):
                        # ans = dset.label2ans[l] if s > 0.5 else 'UQ'
                        # quesid2ans[qid] = ans
                        
                        # notice that we save the score corresponding to maximum score
                        # in contrast to (Siddharth et al. 2021), they use the score corresponding to correct answer
                        # we do this because for UQ, there is no correct answer
                        quesid2score[qid] = (s, evaluator.dataset.id2datum[qid]['label'], dset.label2ans[l])

            # log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)
            log_str = "\nEpoch %d: Train Loss %0.2f\n" % (epoch, loss.item())

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")
            
                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
            
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
    
    def ood_evaluate(self, eval_tuple: DataTuple, dump=None):
        from gqa_data import GQAOODEvaluator
        self.model.eval()
        dset, loader, _ = eval_tuple
        evaluator = GQAOODEvaluator(dset, args.tau)
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
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
    
    def load_teacher(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.teacher.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    # Build Class
    gqa = GQA()

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            gqa.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'testdev' in args.test:
            if 'GQAUQ' in args.test:
                result = gqa.ood_evaluate(
                    get_tuple(args.test, bs=args.batch_size,
                            shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, f'{args.test}_predict.json')
                )
                print(result)
            else:
                result = gqa.evaluate(
                    get_tuple('testdev', bs=args.batch_size,
                            shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'testdev_predict.json')
                )
                print(result)
    elif args.pseudo is not None:
        gqa.get_pseudo_labels(
            get_tuple(args.pseudo, bs=args.batch_size,
                      suhffle=False, drop_last=False),
            dump=os.path.join(args.output, 'pseudo_data.json')
        )
    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            # print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)