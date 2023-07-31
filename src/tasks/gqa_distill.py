"""
select re-assigned questions for training:
This file will predict with large amount of re-assigned visual questions.
1. The ones with high confidence and low variance will be assigned a label by teacher model
and then feed to student model.
2. The ones with low confidence and low variance will be assigned as UQ and then push the student
confidence to zero.
3. After selection, AQ and UQ will be tailored to balance the amount of two.
"""

# TODO:
# 1. sample data from hard negative indices
# 2. implement predict model
# 3. save data according to the original format

import pickle
import json
import random
from tqdm import tqdm
import argparse
import os
import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from utils import load_obj_tsv
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

def parse_args():
    parser = argparse.ArgumentParser()

    # === model ===
    parser.add_argument("--model_dir", default="snap/gqa/gqa_conf_lxr955_chart/chart", type=str,
                        help='path to saved model checkpoints')
    parser.add_argument("--model", default="conf", type=str,
                        help='model name')
    # === data ===
    parser.add_argument("--train_question_file", default="snap/gqa/gqa_conf_lxr955_chart/maps", type=str,
                        help='path to save the plot')
    parser.add_argument("--all_negative_idx_file", default="data/gqa/hardnegative_index.pkl",
                        help="path to all indices of hard negative questions")
    parser.add_argument("--output_name", type=str, required=True)
    # === misc ===
    parser.add_argument("--N", type=int, default=5e6,
                        help="number of examples to be sampled")
    # parser.add_argument("--gamma_a", type=float, default=0.1,
    #                     help="ratio for answerable questions")
    # parser.add_argument("--gamma_u", type=float, default=0.5,
    #                     help="ratio for unanswerable questions")
    parser.add_argument("--tau_aq_c", type=float, default=0.5,
                        help="lower bound on confidence for AQ")
    parser.add_argument("--tau_aq_v", type=float, default=0.15,
                        help="upper bound on variability for AQ")
    parser.add_argument("--tau_uq_c", type=float, default=0.1,
                        help="upper bound on confidence for UQ")
    parser.add_argument("--tau_uq_v", type=float, default=0.05,
                        help="upper bound on variability for UQ")
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--out", default="snap/gqa/distill", type=str)
    parser.add_argument("--balance", action="store_true",
                        help="If you wanna balance your dataset")

    args = parser.parse_args()

    return args

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 20

class GQAModel(torch.nn.Module):
    def __init__(self, args, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        self.logit_fc = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            torch.nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit

class GQADATA(Dataset):
    def __init__(self, data, sampled_data):
        self.data = data

        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }
        self.sampled_data = sampled_data

        # Answers
        self.ans2label = json.load(open("data/gqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/gqa/trainval_label2ans.json"))

        self.ans2label['UQ'] = len(self.ans2label)
        self.label2ans.append('UQ')

        self.num_answers = len(self.ans2label)

        self.splits = ["train"]
    
    def __len__(self):
        return len(self.sampled_data)

class GQABufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, name, number):
        if name == 'testdev':
            path = "data/vg_gqa_imgfeat/gqa_testdev_obj36.tsv"
        else:
            path = "data/vg_gqa_imgfeat/vg_gqa_obj36.tsv"
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]


gqa_buffer_loader = GQABufferLoader()


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class GQATorchDataset(Dataset):
    def __init__(self, dataset: GQADATA):
        super().__init__()
        self.raw_dataset = dataset

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        # only use 5000 images for debug
        img_data.extend(gqa_buffer_loader.load_data('train', -1))
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.sampled_data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        # ques = datum['sent']
        # this is different because we need to re-assign question
        orig_ques_id = datum['original_question_id']
        ques = self.raw_dataset.id2datum[orig_ques_id]['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        return ques_id, feats, boxes, ques

def sample_data_for_distill(args, train_questions, hn_indices):
    imgids = set([q['img_id'] for q in train_questions])
    num_q_each = int(args.N // len(imgids))
    sampled_data = []
    cnt = 0
    print(f"Sample {num_q_each} questions for each image ...")
    for imgid in tqdm(list(imgids)):
        cand_qids = \
            random.sample(hn_indices[imgid], min(num_q_each, len(hn_indices[imgid])))
        for qid in cand_qids:
            sampled_data.append({
                'img_id': imgid,
                'original_question_id': qid,
                'question_id': cnt
            })
            cnt += 1

    print(f"{len(sampled_data)} data sampled!")
    
    return sampled_data

@torch.no_grad()
def get_conf(model, dataloader, sampled_data):
    # modify sampled_data inplace
    # get model confidence
    print("Computing model confidence ...")
    dset = dataloader.dataset.raw_dataset
    for quesid, feats, boxes, sent in tqdm(dataloader):
        feats, boxes = feats.cuda(), boxes.cuda()
        logit = model(feats, boxes, sent)
        # TODO: add more models
        # only support conf model for now
        score, label = torch.sigmoid(logit).max(1)
        for qid , l, s in zip(quesid, label.cpu().numpy(), score.cpu().numpy()):
            if 'confidence' not in sampled_data[qid]:
                sampled_data[qid]['confidence'] = [s]
            else:
                sampled_data[qid]['confidence'].append(s)
            
            if 'answer' not in sampled_data[qid]:
                sampled_data[qid]['answer'] = [dset.label2ans[l]]
            else:
                sampled_data[qid]['answer'].append(dset.label2ans[l])

def calc_stats(sampled_data):
    for i in range(len(sampled_data)):
        if 'confidence' in sampled_data[i]:
            conf_mean = np.mean(sampled_data[i]['confidence'])
            sampled_data[i]['variability'] = \
                np.sqrt(((np.asarray(sampled_data[i]['confidence']) - conf_mean) ** 2).mean())
            sampled_data[i]['confidence'] = conf_mean
            sampled_data[i]['answer'] = Counter(sampled_data[i]['answer']).most_common(1)[0][0]

def filter_data_for_distill(args, train_questions, sampled_data):
    n_orig = len(train_questions)
    conf = np.asarray([d['confidence'] for d in sampled_data])
    var = np.asarray([d['variability'] for d in sampled_data])
    uq_indices = np.where(np.logical_and(conf < args.tau_uq_c, var < args.tau_uq_v))[0]
    aq_indices = np.where(np.logical_and(conf > args.tau_aq_c, var < args.tau_aq_v))[0]
    # uq_indices = np.argsort(uq_score)[:int(args.gamma_u * len(sampled_data))]
    # aq_indices = np.argsort(aq_score)[:int(args.gamma_a * len(sampled_data))]

    aq, uq = [], []
    for ind in uq_indices:
        d = sampled_data[ind]
        d['label'] = {'UQ': 1}
        d['question_id'] = d['img_id'] + '+' + d['original_question_id']
        del d['confidence']
        del d['answer']
        del d['variability']
        uq.append(d)
    for ind in aq_indices:
        d = sampled_data[ind]
        if 'confidence' not in d:
            continue
        d['label'] = {d['answer']: float(d['confidence'])}
        d['question_id'] = d['img_id'] + '+' + d['original_question_id']
        del d['confidence']
        del d['answer']
        del d['variability']
        aq.append(d)
    
    random.shuffle(uq)
    random.shuffle(aq)

    # tailor the amount so that AQ and UQ are balanced
    # (not 1.25 times more than each other)
    if args.balance:
        n_uq, n_aq = len(uq), len(aq)
        if n_aq + n_orig > n_uq * 1.25:
            aq = aq[:int(n_uq * 1.25 - n_orig)]
        elif n_uq > (n_aq + n_orig) * 1.25:
            uq = uq[:int((n_aq + n_orig) * 1.25)]
    print(f"{len(aq)} AQs and {len(uq)} UQs have been created!")
    return aq + uq

def main(args):
    print(args)
    random.seed(0)

    args.from_scratch = False
    args.llayers = 9
    args.xlayers = 5
    args.rlayers = 5

    print(f"Loading training questions from {args.train_question_file} ...")
    with open(args.train_question_file, 'r') as f:
        train_questions = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    output_file = os.path.join(args.out, "conf.pkl")
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            sampled_data = pickle.load(f)
    else:
        print(f"Loading hard negative indices from {args.all_negative_idx_file} ...")
        with open(args.all_negative_idx_file, 'rb') as f:
            hn_idices = pickle.load(f)
        
        print(f"Sampling {args.N} data from all hard negative indices ...")
        sampled_data = sample_data_for_distill(args, train_questions, hn_idices)

        raw_dataset = GQADATA(train_questions, sampled_data)
        torch_dataset = GQATorchDataset(raw_dataset)
        dataloader = DataLoader(
            torch_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=4,
            drop_last=False, pin_memory=True
        )

        model_fns = \
            sorted([fn for fn in os.listdir(args.model_dir) if fn.startswith('EPOCH')])
        model_fns = [os.path.join(args.model_dir, fn) for fn in model_fns]
        model = GQAModel(args, raw_dataset.num_answers - 1)
        model.eval()
        model.cuda()

        for fn in model_fns:
            print(f"Load model from {fn} ...")
            state_dict = torch.load(fn)
            for key in list(state_dict.keys()):
                if '.module' in key:
                    state_dict[key.replace('.module', '')] = state_dict.pop(key)
            model.load_state_dict(state_dict, strict=False)

            get_conf(model, dataloader, sampled_data) # inplace modify sampled_data
        
        print("Calculating statistics ...")
        calc_stats(sampled_data) # inplace modify sampled_data
        
        with open(output_file, 'wb') as f:
            pickle.dump(sampled_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Filtering from {len(sampled_data)} questions ...")
    filtered_data = filter_data_for_distill(args, train_questions, sampled_data)

    # with open(f"data/gqa/{args.output_name}.json", 'w') as f:
    with open(f"{args.output_name}.json", 'w') as f:
        json.dump(filtered_data, f)


if __name__ == "__main__":
    main(parse_args())