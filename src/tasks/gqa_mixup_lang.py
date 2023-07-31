# TreeMix: Compositional Constituency-based Data Augmentation for Natural Language Understanding
# https://arxiv.org/abs/2205.06153
# https://github.com/Magiccircuit/TreeMix/blob/main/Augmentation.py

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
from nltk import Tree
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


def mix_trees(parse1, parse2, lam1, lam2):
    """
    parse1, parse2: constituency parsing from sentences
    lam1, lam2: upper and lower limit of ratio
    """
    assert lam1 > lam2, print(lam1, lam2)
    t1 = Tree.fromstring(parse1)
    original_sentence = ' '.join(t1.leaves()) + '?'
    t1_len = len(t1.leaves())

    # 1. prepare candidates
    candidate_subtree1=list(t1.subtrees(lambda t: lam1>len(t.leaves())/t1_len>lam2))
    t2=Tree.fromstring(parse2)
    candidate_subtree2=list(t2.subtrees(lambda t: lam1>len(t.leaves())/t1_len>lam2))

    # 2. randomly choose spans
    candidate1=random.choice(candidate_subtree1)
    candidate2=random.choice(candidate_subtree2)

    # 3. exchange spans
    exchanged_span=' '.join(candidate1.leaves())
    exchanged_len=len(candidate1.leaves())
    exchanging_span=' '.join(candidate2.leaves())
    new_sentence=original_sentence.replace(exchanged_span,exchanging_span)

    # 4. compute labels
    exchanging_len=len(candidate2.leaves())
    new_len=t1_len-exchanged_len+exchanging_len

    label = (new_len-exchanging_len)/new_len
    return new_sentence, label


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

        if args.backbone == "lxmert":
            self.model = GQAModel(self.train_tuple.dataset.num_answers - 1)
        elif args.backbone == "butd":
            dictionary, emb = gqa_create_dictionary_glove()
            self.model = GQABUTD(self.train_tuple.dataset.num_answers - 1, dictionary)
            self.model.w_emb.load_embeddings(emb)
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
            for i, (ques_id, feats, boxes, sent, parse, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                target = target[:, :-1]

                img_id = [dset.id2datum[qid]['img_id'] for qid in ques_id]
                
                if args.sample_pair:
                    # sample in-batch negative
                    neg_sent, neg_parse, neg_qid, neg_imgid = [], [], [], []
                    for j, qid, in enumerate(ques_id):
                        qid_r = random.choice(ques_id)
                        while dset.id2datum[qid_r]['img_id'] == dset.id2datum[qid]['img_id']:
                            qid_r = random.choice(ques_id)
                        ind_r = ques_id.index(qid_r)
                        neg_sent.append(sent[ind_r])
                        neg_parse.append(parse[ind_r])
                        # positive image + negative question
                        neg_qid.append(dset.id2datum[qid]['img_id'] + '+' + qid_r)
                        # save the negative image
                        neg_imgid.append(dset.id2datum[qid_r]['img_id'])
                    
                    feats = feats.repeat(2, 1, 1)
                    boxes = boxes.repeat(2, 1, 1)
                    target = torch.cat([target, torch.zeros_like(target)], 0)
                    sent = sent + neg_sent
                    parse = parse + neg_parse
                    ques_id = ques_id + neg_qid
                    img_id = img_id + neg_imgid

                if args.mixup_mode.startswith("treemix"):
                    # 1. sample negative sent
                    pet_sent, pet_target = [], []
                    for j, qid in enumerate(ques_id):
                        # the question should have different image id
                        ind_r = random.choice(range(len(ques_id)))
                        while img_id[ind_r] == img_id[j]:
                            ind_r = random.choice(range(len(ques_id)))
                        parse_pos = parse[j]
                        parse_neg = parse[ind_r]
                        # lam1 is upper limit, lam2 is lower limit
                        mixed_sent, label = mix_trees(parse_pos, parse_neg, args.lam1, args.lam2)
                        pet_sent.append(mixed_sent)
                        if args.mixup_mode == 'treemix_v1':
                            pet_target.append(target[j] * label)
                        elif args.mixup_mode == 'treemix_v2':
                            pet_target.append(target[j] * 0)
                        else:
                            raise ValueError(args.mixup_mode)

                    # 2. expand inputs
                    feats = feats.repeat(2, 1, 1)
                    boxes = boxes.repeat(2, 1, 1)
                    target = torch.cat([target, torch.stack(pet_target, 0)], 0)
                    sent = sent + pet_sent
                                
                # 3. forward and compute loss
                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
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