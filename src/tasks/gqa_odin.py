# https://github.com/deeplearning-wisc/multi-label-ood/blob/0c999beb57319016740f19b72bc87c1115c80127/lib.py#L11

import os
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

from param import args
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
        if args.backbone == "lxmert":
            self.model = GQAModel(self.train_tuple.dataset.num_answers - 1)
        elif args.backbone == "butd":
            dictionary, emb = gqa_create_dictionary_glove()
            self.model = GQABUTD(self.train_tuple.dataset.num_answers - 1, dictionary, dropout=False)
            self.model.w_emb.load_embeddings(emb)
        elif args.backbone == "uniter":
            self.model = GQAUNITER(self.train_tuple.dataset.num_answers - 1)
        else:
            raise ValueError(f"Backbone {args.backbone} not implemented!")

        # Load pre-trained weights
        # if args.load_lxmert is not None:
        #     self.model.lxrt_encoder.load(args.load_lxmert)
        # if args.load_lxmert_qa is not None:
        #     load_lxmert_qa(args.load_lxmert_qa, self.model,
        #                    label2ans=self.train_tuple.dataset.label2ans[:-1])

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
            
    def ood_evaluate(self, eval_tuple: DataTuple, dump=None):
        from gqa_data import GQAOODEvaluator
        if args.backbone in ['lxmert', 'uniter']:
            self.model.eval()
        elif args.backbone in ['butd']:
            self.model.train()
        dset, loader, _ = eval_tuple
        evaluator = GQAOODEvaluator(dset)
        quesid2ans = {}
        bceloss = nn.BCEWithLogitsLoss(reduction="none")
        for i, datum_tuple in tqdm(enumerate(loader), total=len(loader)):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            feats, boxes = feats.cuda(), boxes.cuda()

            feats, boxes = Variable(feats, requires_grad=True), Variable(boxes, requires_grad=True)

            logit = self.model(feats, boxes, sent)
            # use temperature scaling
            preds = torch.sigmoid(logit / args.temperature)

            labels = torch.ones(preds.shape).cuda() * (preds >= 0.5)
            labels = Variable(labels.float())

            # input pre-processing
            loss = bceloss(logit, labels)

            idx = torch.max(preds, dim=1)[1].unsqueeze(-1)
            loss = torch.mean(torch.gather(loss, 1, idx))
            
            loss.backward()
            # calculating the perturbation
            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(feats.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            featsPt = torch.add(feats.data, gradient, alpha=-args.noise)

            gradient = torch.ge(boxes.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            boxesPt = torch.add(boxes.data, gradient, alpha=-args.noise)

            with torch.no_grad():
                logit = self.model(
                    Variable(featsPt),
                    Variable(boxesPt),
                    sent
                )

                outputs = torch.sigmoid(logit / args.temperature)
                score, label = outputs.max(1)

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
