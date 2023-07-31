import json
from tqdm import tqdm
import numpy as np
import collections

import torch
from torch.utils.data.dataloader import DataLoader
from tasks.gqa_model import GQAModel
from tasks.gqa_data import GQATorchDataset, GQAEvaluator
from param import args


class GQADataset:
    """
    A GQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """
    def __init__(self, data):
        self.splits = "train"

        self.id2datum = data

        self.data = [v for k, v in self.id2datum.items()]

        # Answers
        self.ans2label = json.load(open("data/gqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/gqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans
        
        self.num_answers = len(self.ans2label)

    # @property
    # def num_answers(self):
    #     return len(self.ans2label)

    def __len__(self):
        return len(self.id2datum)

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_tuple(data, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(data)
    # for gqa-uq we add one more class
    dset.ans2label['UQ'] = dset.num_answers
    dset.label2ans.append('UQ')
    dset.num_answers = len(dset.label2ans)
    print(f"One more answer added. Now there are {dset.num_answers} answers.")
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=4,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


def get_conf(model, quesid2conf, loader):
    model.eval()
    for i, datum_tuple in tqdm(enumerate(loader), total=len(loader)):
        ques_id, feats, boxes, sent = datum_tuple[:4]
        with torch.no_grad():
            feats, boxes = feats.cuda(), boxes.cuda()
            logit = model(feats, boxes, sent)
            scores = torch.sigmoid(logit)

            for qid, s in zip(ques_id, scores.cpu().numpy()):
                if qid in quesid2conf:
                    quesid2conf[qid] *= s
                else:
                    quesid2conf[qid] = s
    return quesid2conf


def main(args):
    print(args)

    data = {}
    for dataset in args.train.split(","):
        with open(f"data/gqa/{dataset}.json", 'r') as f:
            for d in json.load(f):
                if 'UQ' in d['label']:
                    qid = f"{d['unanswerable_reason']}+{d['question_id']}"
                    data[qid] = {
                        'img_id': d['img_id'],
                        'label': d['label'],
                        'sent': d['sent'],
                        'unanswerable_reason': d['unanswerable_reason'],
                        'question_id': qid
                    }
                else:
                    qid = d['question_id']
                    data[qid] = {
                        'img_id': d['img_id'],
                        'label': d['label'],
                        'sent': d['sent'],
                        'question_id': qid
                    }
    print(f"{len(data)} data loaded from {args.train}.")
    
    data_tuple = get_tuple(data, bs=args.batch_size, shuffle=False, drop_last=False)
    dset, loader, _ = data_tuple
    model = GQAModel(dset.num_answers - 1)
    model.cuda()
    quesid2conf = {}
    for path in args.load.split(","):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=False)

        quesid2conf = get_conf(model, quesid2conf, loader)
    
    final_data = []
    for qid in quesid2conf:
        score = quesid2conf[qid]
        label = np.argmax(score)
        ans = dset.label2ans[label]
        final_data.append({
            "img_id": data[qid]['img_id'],
            "question_id": data[qid]['question_id'],
            "sent": data[qid]['sent'],
            "label": {ans: float(score[label])}
        })

    with open(f"data/gqa/{args.output}.json", 'w') as f:
        json.dump(final_data, f)


if __name__ == "__main__":
    main(args)