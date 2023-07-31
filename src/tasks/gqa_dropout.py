# Use the average confidence across ensemble of dropout models

import os
import pickle
from collections import defaultdict, Counter, namedtuple

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator


DataTuple = namedtuple("DataTuple", 'dataset loader evaluator')


def apply_dropout(m):
    # https://worksheets.codalab.org/rest/bundles/0xe921cd3710444eeca7c4d80bf7a0a748/contents/blob/bert_squad.py
    # just set dropout modules to trainable
    if type(m) == nn.Dropout:
        m.train()


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


def get_tuple_uq(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
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
        # self.train_tuple = get_tuple(
        #     args.train, bs=args.batch_size, shuffle=True, drop_last=True
        # )
        # if args.valid != "":
        #     valid_bsize = 2048 if args.multiGPU else 512
        #     self.valid_tuple = get_tuple(
        #         args.valid, bs=valid_bsize,
        #         shuffle=False, drop_last=False
        #     )
        # else:
        #     self.valid_tuple = None
        self.test_tuple = get_tuple_uq(args.test, bs=args.batch_size,
                shuffle=False, drop_last=False)

        self.model = GQAModel(self.test_tuple.dataset.num_answers - 1)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
    
    def predict(self, dump=None):
        from tasks.gqa_data import GQAOODEvaluator
        self.model.eval()
        dset, loader, _ = self.test_tuple
        evaluator = GQAOODEvaluator(dset, args.tau)
        quesid2ans = defaultdict(list)
        quesid2conf = defaultdict(list)
        seed_list = args.seed_list.strip().split(",")
        for seed in seed_list:
            seed = int(seed)
            torch.manual_seed(seed)
            self.model.apply(apply_dropout)
            for i, datum_tuple in tqdm(enumerate(loader), total=len(loader)):
                ques_id, feats, boxes, sent = datum_tuple[:4]
                with torch.no_grad():
                    feats, boxes = feats.cuda(), boxes.cuda()
                    logit = self.model(feats, boxes, sent)
                    score, label = torch.sigmoid(logit).max(1)
                    for qid, l, s in zip(ques_id, label.cpu().numpy(), score.cpu().numpy()):
                        ans = dset.label2ans[l]
                        quesid2ans[qid].append(ans)
                        quesid2conf[qid].append(s)
        quesid2ansconf = {}
        for qid in quesid2ans:
            mean_conf = sum(quesid2conf[qid]) / len(seed_list)
            ans = Counter(quesid2ans[qid]).most_common()[0][0]
            quesid2ansconf[qid] = (ans, mean_conf)

        results = evaluator.evaluate(quesid2ansconf)
        if dump is not None:
            evaluator.dump_result(quesid2ansconf, dump)
        return results

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
            result = gqa.predict(
                dump=os.path.join(args.output, f'{args.test}_predict.json')
            )
            print(result)
        else:
            raise RuntimeError("Can not be applied on datasets w/o unanswerable questions!")
    else:
        raise RuntimeError("Can not be trained!")