import os
import math
import json
import pickle
import collections
import random
import numpy as np
from copy import deepcopy

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from param import args
# from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator, GQADataset_v2

from lxrt.modeling import LXRTPretraining
from lxrt.tokenization import BertTokenizer
from lxrt.entry import convert_sents_to_features

# from butd.preprocess import gqa_create_dictionary_glove
# from butd.butd import GQABUTD

from POSTree import POSTree, check_answer_valid, get_parse_tree_for_batch

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
MAX_STATE_LENGTH = 30

def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    print(f"Parsing {len(dset.id2datum)} sentences ...")
    parsed_data = parse_sents(dset, bs=bs, dump=os.path.join(args.output, f"{splits.replace(',', '')}_converted_v2.json"))
    parsed_dset = GQADataset_v2(parsed_data, splits)
    tset = GQATorchDataset(parsed_dset)
    evaluator = GQAEvaluator(parsed_dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )
    # tset = None
    # evaluator = None
    # data_loader = None

    return DataTuple(dataset=parsed_dset, loader=data_loader, evaluator=evaluator)


def parse_sents(dset, bs, dump=None):
    """
    Constituency parsing demands some time, run the parsing offline and
    save the results for future use.
    """
    if os.path.exists(dump):
        with open(dump, 'r') as f:
            parsed_data = json.load(f)
    else:
        # iter_wrapper = \
        #     (lambda x: tqdm(x, total=int(math.ceil(len(dset.id2datum)/bs)), desc="Parsing")) if args.tqdm else (lambda x: x)

        parsed_data = []
        all_qids = list(dset.id2datum.keys())
        # for i in iter_wrapper(range(0, len(all_qids), bs)):
        #     sents = [dset.id2datum[qid]['sent'] for qid in all_qids[i:i+bs]]
        #     trees, choices = get_parse_tree_for_batch(sents)
        #     for qid, tree, choice in zip(all_qids[i:i+bs], trees, choices):
        #         datum = dset.id2datum[qid]
        #         postree = POSTree(tree, choices)
        #         state = postree.adjust_order()
        #         if state:
        #             new_datum = deepcopy(datum)
        #             new_datum['sq'] = postree.root.first_child.tag == 'SQ'
        #             new_datum['state'] = state
        #             new_datum['choices'] = choice
        #         else:
        #             new_datum = deepcopy(datum)
        #             new_datum['sq'] = postree.root.first_child.tag == 'SQ'
        #             new_datum['state'] = ' '.join([new_datum['sent'], '**blank**'])
        #             new_datum['choices'] = choice
        #         parsed_data.append(new_datum)
        for qid in tqdm(all_qids):
            new_datum = deepcopy(dset.id2datum[qid])
            new_datum['state'] = ' '.join([new_datum['sent'], '**blank**'])
            new_datum['choices'] = []
            new_datum['sq'] = None
            parsed_data.append(new_datum)
        
        if dump is not None:
            with open(dump, 'w') as f:
                json.dump(parsed_data, f)
            print(f"{len(parsed_data)} parsed data have been saved to {dump}!")
        
    return parsed_data


class GQA:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=not args.chart
        )
        if args.valid != "":
            valid_bsize = 512 if args.multiGPU else 128
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        # self.model = LXRTPretraining.from_pretrained(
        #     "bert-base-uncased",
        #     task_mask_lm=True,
        #     task_obj_predict=True,
        #     task_matched=True,
        #     task_qa=False,
        #     visual_losses=args.visual_losses,
        # )
        # self.proc = BertTokenizer.from_pretrained(
        #     "bert-base-uncased",
        #     do_lower_case=True
        # )
        self.model = GQAModel(1)

        # Load pre-trained weights
        print("Loading match model ...")
        if args.load_lxmert is not None:
            # self.load(self.model, args.load_lxmert)
            self.model.lxrt_encoder.load(args.load_lxmert)
        
        self.gqa_model = GQAModel(self.train_tuple.dataset.num_answers)
        print("Loading VQA model ...")
        self.load(self.gqa_model, "snap/gqa/gqa_lxr955/BEST")
        # self.load(self.gqa_model, "snap/gqa/gqa_conf_lxr955/BEST")
        
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
        self.gqa_model.eval()
        dset, loader, _ = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader), desc="Training")) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            for i, (ques_id, feats, boxes, sent, states, _, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                # 1. prepare the inputs
                target = torch.argmax(target, dim=-1)
                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                feats_s = feats.repeat(2, 1, 1) # expand on batch dim
                boxes_s = boxes.repeat(2, 1, 1) # expand on batch dim

                # 2. sample random negative questions in-batch
                indices_r = []
                for qid in ques_id:
                    # the question should have different image id
                    img_id = dset.id2datum[qid]['img_id']
                    qid_r = random.choice(ques_id)
                    while dset.id2datum[qid_r]['img_id'] == img_id:
                        qid_r = random.choice(ques_id)
                    indices_r.append(ques_id.index(qid_r))
                    sent.append(sent[indices_r[-1]])

                # 2. get topk preds
                with torch.no_grad():
                    logits = self.gqa_model(feats_s, boxes_s, sent)
                    proposals = torch.topk(logits, k=args.topk, dim=-1).indices
                
                statements = []
                labels = []
                # 3. get positive statement
                # choices = [c.split(",") if c else [] for c in choices]
                # for j, (targ, state, choice) in enumerate(zip(target, states, choices)):
                for j, (targ, state) in enumerate(zip(target, states)):
                    ans = dset.label2ans[targ.item()]
                    # ans = POSTree.prepare_answer(ans, choice)
                    # assert ans or ans == ''
                    # if not (ans or ans == ''):
                    #     breakpoint()
                    # if ans == '':
                        # statements.append(state.replace(" **blank**", ans))
                    # else:
                        # statements.append(state.replace("**blank**", ans))
                    statements.append(state.replace("**blank**", ans))
                    labels.append(1)
                
                # 4. get negative statement
                # for j, (prop, targ, state, choice) in enumerate(zip(proposals[:len(ques_id)], target, states, choices)):
                for j, (prop, targ, state) in enumerate(zip(proposals[:len(ques_id)], target, states)):
                    # check there are choices
                    # choices exist, return opposite
                    # choices not exist, return sampled
                    ans = dset.label2ans[targ.item()]
                    # if choice:
                    #     ind = check_answer_valid(ans, choice)
                    #     assert type(ind) == int
                    #     prop_r = choice[int(not ind)]
                    # else:
                    #     # we only sample negative answers here
                    #     # if equal to target, prob is set to 0
                    #     weights = (prop != targ).float()
                    #     prop_r = dset.label2ans[prop[torch.multinomial(weights, 1)[0]].item()]
                    weights = (prop != targ).float()
                    prop_r = dset.label2ans[prop[torch.multinomial(weights, 1)[0]].item()]
                    # 3.3 get negative statement
                    # prop_r = POSTree.prepare_answer(prop_r, choice)
                    # assert prop_r or prop_r == ''
                    # if prop_r == '':
                    #     statements.append(state.replace(" **blank**", prop_r))
                    # else:
                    #     statements.append(state.replace("**blank**", prop_r))
                    statements.append(state.replace("**blank**", prop_r))
                    labels.append(0)
                
                # 5. convert random questions into statements
                for j, (prop, ind) in enumerate(zip(proposals[len(ques_id):].cpu().tolist(), indices_r)):
                    # state, choice = states[ind], choices[ind] # use the saved index to retrieve the statements
                    state = states[ind]
                    # if choice: # if you have a choice, then just sample from it
                    #     prop_r = random.choice(choice)
                    # else:
                    #     prop_r = dset.label2ans[random.choice(prop)] # only use one of the proposal
                    prop_r = dset.label2ans[random.choice(prop)]
                    # prop_r = POSTree.prepare_answer(prop_r, choice)
                    # assert prop_r or prop_r == ''
                    # if prop_r == '':
                    #     statements.append(state.replace(" **blank**", prop_r))
                    # else:
                    #     statements.append(state.replace("**blank**", prop_r))
                    statements.append(state.replace("**blank**", prop_r))
                    labels.append(0)

                # 6. prepare inputs for verifier
                # expand the image features
                feats_s = feats.repeat(3, 1, 1) # expand on batch dim
                boxes_s = boxes.repeat(3, 1, 1) # expand on batch dim
                # sent_features = convert_sents_to_features(statements, MAX_STATE_LENGTH, self.proc) # statements are usually longer
                # input_ids = torch.tensor([f.input_ids for f in sent_features], dtype=torch.long).cuda()
                # input_mask = torch.tensor([f.input_mask for f in sent_features], dtype=torch.long).cuda()
                # segment_ids = torch.tensor([f.segment_ids for f in sent_features], dtype=torch.long).cuda()
                labels = torch.tensor(labels, dtype=torch.float).view(-1, 1).cuda()
                logits_ver = self.model(feats_s, boxes_s, statements)

                # 7. compute loss and backprop
                # loss, _, _ = self.model(input_ids, segment_ids, input_mask, visual_feats=feats, pos=boxes, matched_label=labels)
                loss = self.bce_loss(logits_ver, labels)
                # loss = loss * logits_ver.size(1)
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

            log_str = "\nEpoch %d: Train Loss %0.2f\n" % (epoch, loss.item())

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid Accuracy %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best Accuracy %0.2f\n" % (epoch, best_valid * 100.)
            
            if args.save_all:
                self.save(f"EPOCH_{epoch}")

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()
            
        self.save("LAST")
    
    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """
        check the accuracy
        """
        self.gqa_model.eval()
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader), desc="Eval")) if args.tqdm else (lambda x: x)
        quesid2ans = {}
        recall, cnt = 0, 0
        for i, datum_tuple in iter_wrapper(enumerate(loader)):
            ques_id, feats, boxes, sent, states, _ = datum_tuple[:6]
            # choices = [c.split(",") if c else [] for c in choices]

            with torch.no_grad():
                # 1. get topk preds
                feats, boxes = feats.cuda(), boxes.cuda()
                logits = self.gqa_model(feats, boxes, sent)
                proposals = torch.topk(logits, k=args.topk, dim=-1).indices

                # 2. get statements
                statements = []
                answers = []
                # for j, (prop, state, choice) in enumerate(zip(proposals, states, choices)):
                for j, (prop, state) in enumerate(zip(proposals, states)):
                    # if not choice:
                    #     prop = [dset.label2ans[p.item()] for p in prop]
                    # else:
                    #     # you can only choose from choices
                    #     prop = [choice[0]] + [choice[1]] * (args.topk - 1)
                    prop = [dset.label2ans[p.item()] for p in prop]
                    answers.extend(prop)
                    for p in prop:
                        # p = POSTree.prepare_answer(p, choice)
                        # assert p or p == ''
                        # if p == '':
                        #     statements.append(state.replace(" **blank**", p))
                        # else:
                        #     statements.append(state.replace("**blank**", p))
                        statements.append(state.replace("**blank**", p))
                
                # 3. prepare inputs for verifier
                feats_s = []
                boxes_s = []
                for fe, bo in zip(feats, boxes):
                    feats_s.append(fe.repeat(args.topk, 1, 1))
                    boxes_s.append(bo.repeat(args.topk, 1, 1))
                feats_s = torch.cat(feats_s, 0)
                boxes_s = torch.cat(boxes_s, 0)
                # sent_features = convert_sents_to_features(statements, MAX_STATE_LENGTH, self.proc)
                # input_ids = torch.tensor([f.input_ids for f in sent_features], dtype=torch.long).cuda()
                # input_mask = torch.tensor([f.input_mask for f in sent_features], dtype=torch.long).cuda()
                # segment_ids = torch.tensor([f.segment_ids for f in sent_features], dtype=torch.long).cuda()

                # 4. foward
                # score = self.model.forward_match(input_ids, segment_ids, input_mask, feats, boxes)
                score = torch.sigmoid(self.model(feats_s, boxes_s, statements)).squeeze()
                # score = torch.softmax(score.view(-1, 2), dim=-1)[:, -1]
                score = score.cpu().numpy()

                # 5. get answer and confidence for each example
                for qid, j in zip(ques_id, range(0, len(statements), args.topk)):
                    ind = np.argmax(score[j:j+args.topk])
                    ans = answers[j:j+args.topk][ind]
                    quesid2ans[qid] = ans

                    if score[j:j+args.topk][ind] > args.tau:
                        recall += 1
                    cnt += 1
                    
        results = evaluator.evaluate(quesid2ans)
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        print(f"Valid Recall: {recall*100/cnt:.2f}")
        return results
    
    def ood_evaluate(self, eval_tuple: DataTuple, dump=None):
        """
        the max score within candidates will be used as confidence 
        """
        from gqa_data import GQAOODEvaluator
        self.gqa_model.eval()
        self.model.eval()
        dset, loader, _ = eval_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader), desc="OOD Eval")) if args.tqdm else (lambda x: x)
        evaluator = GQAOODEvaluator(dset, args.tau)
        quesid2ans = {}
        for i, datum_tuple in iter_wrapper(enumerate(loader)):
            ques_id, feats, boxes, sent, states, _ = datum_tuple[:6]   # avoid handling target
            # choices = [c.split(",") if c else [] for c in choices]
            with torch.no_grad():
                # 1. get topk preds
                feats, boxes = feats.cuda(), boxes.cuda()
                logits = self.gqa_model(feats, boxes, sent)
                proposals = torch.topk(logits, k=args.topk, dim=-1).indices

                # 2. get statements
                statements = []
                answers = []
                # for j, (prop, state, choice) in enumerate(zip(proposals, states, choices)):
                for j, (prop, state) in enumerate(zip(proposals, states)):
                    # if not choice:
                    #     prop = [dset.label2ans[p.item()] for p in prop]
                    # else:
                    #     # you can only choose from choices
                    #     prop = [choice[0]] + [choice[1]] * (args.topk - 1)
                    prop = [dset.label2ans[p.item()] for p in prop]
                    answers.extend(prop)
                    for p in prop:
                        # p = POSTree.prepare_answer(p, choice)
                        # assert p or p == ''
                        # if p == '':
                        #     statements.append(state.replace(" **blank**", p))
                        # else:
                        #     statements.append(state.replace("**blank**", p))
                        statements.append(state.replace("**blank**", p))
                
                # 3. prepare inputs for verifier
                feats_s = []
                boxes_s = []
                for fe, bo in zip(feats, boxes):
                    feats_s.append(fe.repeat(args.topk, 1, 1))
                    boxes_s.append(bo.repeat(args.topk, 1, 1))
                feats_s = torch.cat(feats_s, 0)
                boxes_s = torch.cat(boxes_s, 0)
                # sent_features = convert_sents_to_features(statements, MAX_STATE_LENGTH, self.proc)
                # input_ids = torch.tensor([f.input_ids for f in sent_features], dtype=torch.long).cuda()
                # input_mask = torch.tensor([f.input_mask for f in sent_features], dtype=torch.long).cuda()
                # segment_ids = torch.tensor([f.segment_ids for f in sent_features], dtype=torch.long).cuda()

                # 4. foward
                # score = self.model.forward_match(input_ids, segment_ids, input_mask, feats, boxes)
                score = torch.sigmoid(self.model(feats_s, boxes_s, statements)).squeeze()
                score = score.cpu().numpy()

                # 5. get answer and confidence for each example
                for qid, j in zip(ques_id, range(0, len(statements), args.topk)):
                    ind = np.argmax(score[j:j+args.topk])
                    ans = answers[j:j+args.topk][ind]
                    conf = score[j:j+args.topk][ind]
                    quesid2ans[qid] = (ans, conf)
        results = evaluator.evaluate(quesid2ans)
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return results

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, _, _, target) in enumerate(loader):
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
            else:
                result = gqa.evaluate(
                    get_tuple('testdev', bs=args.batch_size,
                            shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'testdev_predict.json')
                )
                print(result)
    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)