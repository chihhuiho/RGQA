# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ood_model', action="store_true")
    parser.add_argument('--freeze_vqa_branch', action="store_true",
                        help="applies for branched network only")
    parser.add_argument('--chart', action="store_true",
                        help="save all the scores for creating Dataset Cartography")
    parser.add_argument('--tau', type=float, default=0.5,
                        help="threshold on deciding ood")
    parser.add_argument('--m_in', type=float, default=25.,
                        help="margin for in-distribution; above this value will be penalized")
    parser.add_argument("--m_out", type=float, default=0.,
                        help="margin for out-distribution; below this value will be penalized")
    parser.add_argument("--seed_list", type=str,
                        help="a list of seed, used for test-time dropout")
    parser.add_argument("--save_all", action="store_true",
                        help="save result at each epoch")
    parser.add_argument("--temperature", default=1., type=float,
                        help="temperature scaling for ODIN")
    parser.add_argument("--noise", default=0., type=float,
                        help="noise for perturbation in ODIN")
    parser.add_argument("--mix_branched_score", action="store_true",
                        help="mix scores between 2 branches in branched model")
    parser.add_argument("--project_size", default=128, type=int)
    parser.add_argument("--pseudo", default=None)
    parser.add_argument("--backbone", default="lxmert", type=str)
    parser.add_argument("--ensemble_method", default="mean", type=str,
                        help="Choose from mean or multiply")
    parser.add_argument("--topk", type=int, default=5,
                        help="topk proposals from gqa model")
    parser.add_argument("--mixup_mode", type=str, default='mixup_v1')
    parser.add_argument("--lam1", type=float, default=0.5)
    parser.add_argument("--lam2", type=float, default=0.05)
    parser.add_argument("--sample_pair", action='store_const', default=False, const=True)
    parser.add_argument("--update_weight_model", action='store_const', default=False, const=True)
    parser.add_argument("--teacher_path", default=None, help="path to teacher model")
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--mixup_alpha", type=float, default=1.0)
    parser.add_argument("--mixup_beta", type=float, default=1.0)
    parser.add_argument("--load_gqa", type=str)
    parser.add_argument("--target_acc", type=float, default=None)
    parser.add_argument("--predict", action='store_const', default=False, const=True)
    parser.add_argument("--predict_closeset", action='store_const', default=False, const=True)
    parser.add_argument("--train_pos", default='train')
    parser.add_argument("--train_neg", default="train")
    parser.add_argument('--lr_w', type=float, default=100)

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=0)

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
