# https://github.com/deeplearning-wisc/multi-label-ood/blob/0c999beb57319016740f19b72bc87c1115c80127/lib.py#L178

import os
import pickle
import collections

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

# from lxrt.entry import LXRTEncoder, convert_sents_to_features
# from lxrt.modeling import BertLayerNorm, GeLU

from param import args
# from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel_maha as GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator
from uniter.uniter import GQAUNITER_maha as GQAUNITER
from butd.preprocess import gqa_create_dictionary_glove
from butd.butd import GQABUTD_maha as GQABUTD
import json


# from tasks.gqa_model import MAX_GQA_LENGTH

# class GQAModel(nn.Module):
#     def __init__(self, num_answers):
#         super().__init__()
#         self.lxrt_encoder = LXRTEncoder(
#             args,
#             max_seq_length=MAX_GQA_LENGTH
#         )
#         hid_dim = self.lxrt_encoder.dim
#         self.logit_fc = nn.Sequential(
#             nn.Linear(hid_dim, hid_dim * 2),
#             GeLU(),
#             BertLayerNorm(hid_dim * 2, eps=1e-12),
#             nn.Linear(hid_dim * 2, num_answers)
#         )
#         self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
    
#     def forward(self, feat, pos, input_ids, token_type_ids, attention_mask):
#         """
#         directly take features as input
#         """
#         x = self.lxrt_encoder.model(input_ids, token_type_ids, attention_mask,
#                                     visual_feats=(feat,pos))
#         logit = self.logit_fc(x)
#         return logit, x


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
        # self.model = GQAModel(self.train_tuple.dataset.num_answers - 1)
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
    
    def sample_estimator(self):
        # https://github.com/deeplearning-wisc/multi-label-ood/blob/0c999beb57319016740f19b72bc87c1115c80127/lib.py#L63
        output_file = os.path.join(args.output, "sample_estimates.pkl")
        if os.path.exists(output_file):
            with open(output_file, 'rb') as f:
                data = pickle.load(f)
                sample_class_mean = data['mean']
                precision = data['precision']
        else:
            import sklearn.covariance
            num_classes = self.train_tuple.dataset.num_answers - 1

            group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
            num_sample_per_class = np.zeros((num_classes))
            list_features = [0 for i in range(num_classes)]

            self.model.eval()
            _, loader, _ = self.train_tuple
            with torch.no_grad():
                for i, datum_tuple in tqdm(enumerate(loader), total=len(loader)):
                    _, feats, boxes, sent, target = datum_tuple
                    target = target[:, :-1]
                    
                    feats, boxes = feats.cuda(), boxes.cuda()

                    # sent_features = convert_sents_to_features(
                    #     sent, self.model.lxrt_encoder.max_seq_length,
                    #     self.model.lxrt_encoder.tokenizer
                    # )

                    # input_ids = torch.tensor([f.input_ids for f in sent_features], dtype=torch.long).cuda()
                    # input_mask = torch.tensor([f.input_mask for f in sent_features], dtype=torch.long).cuda()
                    # segment_ids = torch.tensor([f.segment_ids for f in sent_features], dtype=torch.long).cuda()

                    # _, outputs = self.model(feats, boxes, input_ids, segment_ids, input_mask)
                    _, outputs = self.model(feats, boxes, sent)
                    outputs = outputs.cpu()
                    
                    for i in range(target.size(0)):
                        for j in range(num_classes):
                            if target[i][j] == 0:
                                continue
                            label = j
                            if num_sample_per_class[label] == 0:
                                list_features[label] = outputs[i].view(1, -1)
                            else:
                                list_features[label] = torch.cat([list_features[label],
                                                                outputs[i].view(1, -1)], 0)
                            num_sample_per_class[label] += 1
                    
                sample_class_mean = torch.zeros((num_classes, outputs.size(1)))
                for j in range(num_classes):
                    if isinstance(list_features[j], torch.Tensor):
                        sample_class_mean[j] = torch.mean(list_features[j], 0)
                
                X = []
                for i in range(num_classes):
                    if isinstance(list_features[i], torch.Tensor):
                        X.append(list_features[i] - sample_class_mean[i])
                X = torch.cat(X, 0)

                # find inverse
                group_lasso.fit(X.numpy())
                precision = group_lasso.precision_
            
            with open(output_file, 'wb') as f:
                pickle.dump({
                    "mean": sample_class_mean, "precision": precision
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        return sample_class_mean, precision

    def ood_evaluate(self, eval_tuple: DataTuple, pack, dump=None):
        from gqa_data import GQAOODEvaluator
        sample_mean, precision = pack
        if not isinstance(sample_mean, torch.Tensor):
            sample_mean = torch.from_numpy(sample_mean).float().cuda()
        else:
            sample_mean = sample_mean.float().cuda()
        if not isinstance(precision, torch.Tensor):
            precision = torch.from_numpy(precision).float().cuda()
        else:
            precision = precision.float().cuda()
        num_classes = self.train_tuple.dataset.num_answers - 1
        if args.backbone in ['lxmert', 'uniter']:
            self.model.eval()
        elif args.backbone == 'butd':
            self.model.train()
        dset, loader, _ = eval_tuple
        evaluator = GQAOODEvaluator(dset)
        quesid2ans = {}
        for i, datum_tuple in tqdm(enumerate(loader), total=len(loader)):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            feats, boxes = feats.cuda(), boxes.cuda()

            # sent_features = convert_sents_to_features(
            #     sent, self.model.lxrt_encoder.max_seq_length,
            #     self.model.lxrt_encoder.tokenizer
            # )

            # input_ids = torch.tensor([f.input_ids for f in sent_features], dtype=torch.long).cuda()
            # input_mask = torch.tensor([f.input_mask for f in sent_features], dtype=torch.long).cuda()
            # segment_ids = torch.tensor([f.segment_ids for f in sent_features], dtype=torch.long).cuda()

            feats, boxes = Variable(feats, requires_grad=True), Variable(boxes, requires_grad=True)
            # _, outputs = self.model(feats, boxes, input_ids, segment_ids, input_mask)
            _, outputs = self.model(feats, boxes, sent)

            # compute Mahalanobis score
            gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[i]
                zero_f = outputs.data - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                if i == 0:
                    gaussian_score = term_gau.view(-1, 1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
            
            # Input_processing
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = sample_mean.index_select(0, sample_pred)
            zero_f = outputs - Variable(batch_sample_mean)
            pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision)), zero_f.t()).diag()
            loss = torch.mean(-pure_gau)
            loss.backward()

            gradient = torch.ge(feats.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            featsPt = torch.add(feats.data, gradient, alpha=-args.noise)

            gradient = torch.ge(boxes.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            boxesPt = torch.add(boxes.data, gradient, alpha=-args.noise)

            with torch.no_grad():
                _, noised_outputs = self.model(
                    Variable(featsPt),
                    Variable(boxesPt),
                    sent
                )
                noised_gaussian_score = 0
                for i in range(num_classes):
                    batch_sample_mean = sample_mean[i]
                    zero_f = noised_outputs.data - batch_sample_mean
                    term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                    if i == 0:
                        noised_gaussian_score = term_gau.view(-1, 1)
                    else:
                        noised_gaussian_score = torch.cat((noised_gaussian_score, term_gau.view(-1, 1)), 1)
                
                score, label = torch.max(noised_gaussian_score, dim=1)

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
            pack = gqa.sample_estimator()
            result = gqa.ood_evaluate(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                pack,
                dump=os.path.join(args.output, f'{args.test}_predict.json')
            )
            print(result)
            with open(os.path.join(args.output, f'{args.test}_result.json'), "w") as outfile:
                json.dump(result, outfile)
 

        else:
            raise RuntimeError("Can not be applied on datasets w/o unanswerable questions!")
    else:
        raise RuntimeError("Can not be trained!")
