import os
import json
import pickle
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
from caps.caps import GQABERT
from gqa_data import GQADataset, GQACaptionDataset, GQAEvaluator
from uniter.uniter import GQAUNITER
from butd.preprocess import gqa_create_dictionary_glove
from butd.butd import GQABUTD

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
        tset = GQACaptionDataset(dset)
        data_loader = DataLoader(
            tset, batch_size=bs,
            shuffle=shuffle, num_workers=args.num_workers,
            drop_last=drop_last, pin_memory=True
        )
    evaluator = GQAEvaluator(dset)
    

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class GQA:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=not args.chart,
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
        self.model = GQABERT()
        if args.backbone == "lxmert":
            self.gqa_model = GQAModel(self.train_tuple.dataset.num_answers - 1)
        elif args.backbone == "butd":
            dictionary, emb = gqa_create_dictionary_glove()
            self.gqa_model = GQABUTD(self.train_tuple.dataset.num_answers - 1, dictionary)
            self.gqa_model.w_emb.load_embeddings(emb)
        elif args.backbone == 'uniter':
            self.gqa_model = GQAUNITER(self.train_tuple.dataset.num_answers - 1)

        # GPU options
        self.model = self.model.cuda()
        self.gqa_model = self.gqa_model.cuda()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
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

        for epoch in range(args.epochs):
            # quesid2ans = {}
            quesid2score = {}
            for i, (ques_id, _, _, caps, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                target = 1 - target[:, -1]

                target = target.unsqueeze(-1).cuda()
                logit = self.model(caps, sent)

                loss = self.bce_loss(logit, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
            
            # log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)
            log_str = "\nEpoch %d: Train Loss %0.2f\n" % (epoch, loss.item())
            
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
    
    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))
    
    def ood_evaluate(self, eval_tuple: DataTuple, dump=None):
        from gqa_data import GQAOODEvaluator
        self.model.eval()
        self.gqa_model.eval()
        dset, loader, _ = eval_tuple
        evaluator = GQAOODEvaluator(dset)
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, caps, sent = datum_tuple[:5]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                gqa_logit = self.gqa_model(feats, boxes, sent)
                logit = self.model(caps, sent)
                _, label = torch.sigmoid(gqa_logit).max(1)

                score = torch.sigmoid(logit)
                for qid, l, s in zip(ques_id, label.cpu().numpy(), score.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = (ans, s)
        results = evaluator.evaluate(quesid2ans)
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return results
    
    def load(self, path, model):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    # Build Class
    gqa = GQA()

    # Load Model
    if args.load is not None:
        gqa.load(args.load, gqa.model)
    if args.load_gqa is not None:
        gqa.load(args.load_gqa, gqa.gqa_model)
    
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
                result = gqa.ood_evaluate(
                    get_tuple(args.test, bs=args.batch_size,
                            shuffle=False, drop_last=False,
                            vilt=args.backbone == 'vilt'),
                    dump=os.path.join(args.output, f'{args.test}_predict.json')
                )
                print(result)

                with open(os.path.join(args.output, f'{args.test}_result.json'), "w") as outfile:
                    json.dump(result, outfile)
 

            else:
                result = gqa.evaluate(
                    get_tuple('testdev', bs=args.batch_size,
                            shuffle=False, drop_last=False,
                            vilt=args.backbone == 'vilt'),
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
            # print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)
