"""
train a separate model for UQ detection
The model is initialized with pre-trained LXMERT.
And then the accepted instances will be passed to GQA model for prediction.
"""

import os
import collections
import pickle

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from param import args
# from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel, MAX_GQA_LENGTH
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator
import json
# from lxrt.modeling import LXRTPretraining
# from lxrt.tokenization import BertTokenizer
# from lxrt.entry import convert_sents_to_features


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

        # self.model = LXRTPretraining.from_pretrained(
        #     "bert-base-uncased",
        #     task_mask_lm=False,
        #     task_obj_predict=False,
        #     task_matched=True,
        #     task_qa=False,
        # )
        # self.proc = BertTokenizer.from_pretrained(
        #     "bert-base-uncased",
        #     do_lower_case=True
        # )
        self.model = GQAModel(1)

        # Load pre-trained weights
        print("Loading match model ...")
       

        
        if args.load_lxmert is not None:
            # self.load(self.model.lxrt_encoder, args.load_lxmert)
            self.model.lxrt_encoder.load(args.load_lxmert)
        
        

        self.gqa_model = GQAModel(self.train_tuple.dataset.num_answers - 1)
        print("Loading VQA model ...")
        self.load(self.gqa_model, "/data8/srip22vg/projects/vg/reference/lxmert/snap/gqa/gqa_lxr955/BEST")
        
        # GPU options
        self.model = self.model.cuda()
        self.gqa_model = self.gqa_model.cuda()
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
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        if args.chart:
            os.makedirs(os.path.join(args.output, 'chart'), exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            quesid2score = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                target = (target[:, -1] == 0).unsqueeze(-1).float()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                # sent_features = convert_sents_to_features(sent, MAX_GQA_LENGTH, self.proc)
                # input_ids = torch.tensor([f.input_ids for f in sent_features], dtype=torch.long).cuda()
                # input_mask = torch.tensor([f.input_mask for f in sent_features], dtype=torch.long).cuda()
                # segment_ids = torch.tensor([f.segment_ids for f in sent_features], dtype=torch.long).cuda()

                # loss, _, _ = self.model(input_ids, segment_ids, input_mask, visual_feats=feats, pos=boxes, matched_label=target)
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

            log_str = "\nEpoch %d: Train Loss %0.2f\n" % (epoch, loss.item())

            # if self.valid_tuple is not None:  # Do Validation
            #     valid_score = self.evaluate(eval_tuple)
            #     if valid_score > best_valid:
            #         best_valid = valid_score
            #         self.save("BEST")

            #     log_str += "Epoch %d: Valid Recall %0.2f\n" % (epoch, valid_score * 100.) + \
            #                "Epoch %d: Best Recall %0.2f\n" % (epoch, best_valid * 100.)
            
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
    
    def evaluate(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        recall = 0
        cnt = 0
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                # sent_features = convert_sents_to_features(sent, MAX_GQA_LENGTH, self.proc)
                # input_ids = torch.tensor([f.input_ids for f in sent_features], dtype=torch.long).cuda()
                # input_mask = torch.tensor([f.input_mask for f in sent_features], dtype=torch.long).cuda()
                # segment_ids = torch.tensor([f.segment_ids for f in sent_features], dtype=torch.long).cuda()

                # score = self.model.forward_match(input_ids, segment_ids, input_mask, feats, boxes)
                # score = torch.softmax(score.view(-1, 2), dim=-1)[:, -1]
                score = torch.sigmoid(self.model(feats, boxes, sent))

                recall += (score > args.tau).sum().item()
                cnt += len(sent)
        return recall / cnt
    
    def ood_evaluate(self, eval_tuple: DataTuple, dump=None):
        from gqa_data import GQAOODEvaluator
        self.model.eval()
        self.gqa_model.eval()
        dset, loader, _ = eval_tuple
        evaluator = GQAOODEvaluator(dset, args.tau)
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                # sent_features = convert_sents_to_features(sent, MAX_GQA_LENGTH, self.proc)
                # input_ids = torch.tensor([f.input_ids for f in sent_features], dtype=torch.long).cuda()
                # input_mask = torch.tensor([f.input_mask for f in sent_features], dtype=torch.long).cuda()
                # segment_ids = torch.tensor([f.segment_ids for f in sent_features], dtype=torch.long).cuda()
                # logit = self.model.forward_match(input_ids, segment_ids, input_mask, feats, boxes)
                logit = self.model(feats, boxes, sent)
                # score2 = torch.softmax(logit.view(-1, 2), dim=-1)[:, -1]
                score2 = torch.sigmoid(logit).squeeze()

                gqa_logit = self.gqa_model(feats, boxes, sent)
                score1, label = torch.sigmoid(gqa_logit).max(1)
                if args.mix_branched_score:
                    score = score1 * score2
                else:
                    score = score2
                for qid, l, s in zip(ques_id, label.cpu().numpy(), score.cpu().numpy()):
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

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, model, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
            if 'module.' in key:
                state_dict[key.replace('module.', '')] = state_dict.pop(key)
        
        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        model.load_state_dict(state_dict, strict=False)
    

if __name__ == "__main__":
    # Build Class
    gqa = GQA()

    # Load Model
    if args.load is not None:
        gqa.load(gqa.model, args.load)

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
                with open(os.path.join(args.output, f'{args.test}_result.json'), "w") as outfile:
                    json.dump(result, outfile)
 
            else:
                result = gqa.evaluate(
                    get_tuple('testdev', bs=args.batch_size,
                            shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'testdev_predict.json')
                )
                print(result)
                with open(os.path.join(args.output, f'{args.test}_result.json'), "w") as outfile:
                    json.dump(result, outfile)
 

    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)
