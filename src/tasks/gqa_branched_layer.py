import os
import collections
import pickle

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from lxrt.modeling import BertPreTrainedModel, VisualFeatEncoder, VISUAL_CONFIG
from lxrt.modeling import BertLayer, LXRTXLayer, BertEmbeddings, BertPooler
from lxrt.modeling import BertLayerNorm, GeLU
from lxrt.entry import convert_sents_to_features, set_visual_config
from lxrt.tokenization import BertTokenizer

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
# from tasks.gqa_model_branched import GQAModel_branched
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 20


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


class GQAProjector(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, args.project_size),
            nn.Tanh()
        )

    def forward(self, hidden_states):
        # always extract first token
        first_token_tensor = hidden_states[:, 0]
        return self.proj(first_token_tensor)


class LXRTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = VisualFeatEncoder(config)

        # Number of layers
        self.num_l_layers = VISUAL_CONFIG.l_layers
        self.num_x_layers = VISUAL_CONFIG.x_layers
        self.num_r_layers = VISUAL_CONFIG.r_layers
        print("LXRT encoder with %d l_layers, %d x_layers, and %d r_layers." %
              (self.num_l_layers, self.num_x_layers, self.num_r_layers))

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        self.x_layers = nn.ModuleList(
            [LXRTXLayer(config) for _ in range(self.num_x_layers)]
        )
        self.r_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        )

    def forward(self, lang_feats, lang_attention_mask,
                visn_feats, visn_attention_mask=None):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        visn_feats = self.visn_fc(visn_feats)

        # Run language layers
        for layer_module in self.layer:
            lang_feats = layer_module(lang_feats, lang_attention_mask)

        # Run relational layers
        for layer_module in self.r_layers:
            visn_feats = layer_module(visn_feats, visn_attention_mask)

        # Run cross-modality layers
        all_feats = []
        for layer_module in self.x_layers:
            lang_feats, visn_feats = layer_module(lang_feats, lang_attention_mask,
                                                  visn_feats, visn_attention_mask)
            all_feats.append(lang_feats)

        return all_feats


class LXRTModel(BertPreTrainedModel):
    """
    output multiple levels of features as input to detect OOD
    """
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = LXRTEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                visual_feats=None, visual_attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Run LXRT backbone
        all_feats = self.encoder(
            embedding_output,
            extended_attention_mask,
            visn_feats=visual_feats,
            visn_attention_mask=extended_visual_attention_mask)
        pooled_output = self.pooler(all_feats[-1])

        return all_feats, pooled_output


class LXRTFeatureExtraction(BertPreTrainedModel):
    """
    BERT model for classification.
    """
    def __init__(self, config, mode='lxr'):
        """

        :param config:
        :param mode:  Number of visual layers
        """
        super().__init__(config)
        self.bert = LXRTModel(config)
        self.mode = mode
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None,
                visual_attention_mask=None):
        all_feats, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                            visual_feats=visual_feats,
                                            visual_attention_mask=visual_attention_mask)
        if 'x' == self.mode:
            return pooled_output
        elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
            return all_feats, pooled_output
        elif 'l' in self.mode or 'r' in self.mode:
            return all_feats


class LXRTEntry(nn.Module):
    def __init__(self, args, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length
        set_visual_config(args)

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Build LXRT Model
        self.model = LXRTFeatureExtraction.from_pretrained(
            "bert-base-uncased",
            mode='lxr'
        )

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)
            
    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768
    
    def forward(self, sents, feats, visual_attention_mask=None):
        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        all_feats, output = self.model(input_ids, segment_ids, input_mask,
                            visual_feats=feats,
                            visual_attention_mask=visual_attention_mask)
        return all_feats, output

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)


class GQAModel_branched(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEntry(
            args,
            max_seq_length=MAX_GQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        # Build projectors
        model_config = self.lxrt_encoder.model.config
        self.projs = nn.ModuleList(
            [GQAProjector(model_config, args) for _ in range(VISUAL_CONFIG.x_layers)]
        )
        # Build detector
        inp_size = VISUAL_CONFIG.x_layers * args.project_size
        self.conf_fc = nn.Sequential(
            nn.Linear(inp_size, inp_size * 2),
            GeLU(),
            BertLayerNorm(inp_size * 2, eps=1e-12),
            nn.Linear(inp_size * 2, 1)
        )
        # Build classifier
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        self.conf_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        self.projs.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        all_feats, output = self.lxrt_encoder(sent, (feat, pos))
        proj_feats = []
        for proj, feat in zip(self.projs, all_feats):
            proj_feats.append(proj(feat))
        conf = self.conf_fc(torch.cat(proj_feats, 1))
        logit = self.logit_fc(output)

        return logit, conf


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
        
        # there is no need to add extra class
        self.model = GQAModel_branched(self.train_tuple.dataset.num_answers - 1)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans[:-1])
        
        # train with vqa branch and backbone fixed
        if args.freeze_vqa_branch:
            for name, param in self.model.named_parameters():
                if not name.startswith('projs') and\
                    not name.startswith('conf_fc'):
                    param.requires_grad = False
        
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
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)
        
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
            quesid2ans = {}
            quesid2score = {}

            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                id_inds = target[:, -1] == 0
                target = target[:, :-1]

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit, conf = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                if args.mce_loss:
                    max_value, target = target.max(1)
                    loss_ce = self.mce_loss(logit[id_inds], target[id_inds]) * logit.size(1)
                else:
                    loss_ce = self.bce_loss(logit[id_inds], target[id_inds])
                    loss_ce = loss_ce * logit.size(1)
                
                conf = torch.sigmoid(conf).squeeze()
                conf_target = id_inds.float().cuda()
                loss_conf = F.binary_cross_entropy(conf, conf_target)

                loss = loss_ce + loss_conf
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l, c in zip(ques_id, label.cpu().numpy(), conf.cpu().detach().numpy()):
                    ans = dset.label2ans[l] if c > args.tau else 'UQ'
                    quesid2ans[qid] = ans
                    if args.chart:
                        quesid2score[qid] = (c, evaluator.dataset.id2datum[qid]['label'], dset.label2ans[l])
            
            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

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
                logit, conf = self.model(feats, boxes, sent)
                _, label = logit.max(1)
                score = torch.sigmoid(conf)
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
                logit, conf = self.model(feats, boxes, sent)
                score1, label = torch.sigmoid(logit).max(1)
                score2 = torch.sigmoid(conf).squeeze()
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
        print(f"Loading model from {args.load}")
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
    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)