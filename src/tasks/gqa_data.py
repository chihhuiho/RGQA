# coding=utf-8
# Copyleft 2019 project LXRT.

from copy import deepcopy
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv
from vilt.transforms import keys_to_transforms

from PIL import Image

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


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
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/gqa/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

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
        return len(self.data)


class GQADataset_v2:
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
    def __init__(self, data, splits:str):
        # Loading datasets to data
        self.splits = splits.split(',')
        self.data = data
        print("Load %d data." % len(self.data))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

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
        return len(self.data)


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
    def __init__(self, dataset: GQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        load_testdev = all(['testdev' in s for s in dataset.splits])
        if load_testdev:
            img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
        else:
            img_data.extend(gqa_buffer_loader.load_data('train', topk))
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
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
        if 'sent' in datum:
            ques = datum['sent']
        else:
            ques = \
                self.raw_dataset.id2datum[datum['original_question_id']]['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        if args.backbone in ['uniter']:
            boxes = self._uniterBoxes(boxes)
        else:
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)

        # Create target
        if "state" in datum and "choices" in datum:
            state = datum['state']
            if datum['choices']:
                choices = ','.join([' '.join(c) for c in datum['choices']])
            else:
                choices = ''
            if 'label' in datum:
                label = datum['label']
                target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in label.items():
                    if ans in self.raw_dataset.ans2label:
                        target[self.raw_dataset.ans2label[ans]] = score
                return ques_id, feats, boxes, ques, state, choices, target
            else:
                return ques_id, feats, boxes, ques, state, choices
        elif 'parse' in datum:
            parse = datum['parse']
            if 'label' in datum:
                label = datum['label']
                target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in label.items():
                    if ans in self.raw_dataset.ans2label:
                        target[self.raw_dataset.ans2label[ans]] = score
                return ques_id, feats, boxes, ques, parse, target
            else:
                return ques_id, feats, boxes, ques, parse
        else:
            if 'label' in datum:
                label = datum['label']
                target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in label.items():
                    if ans in self.raw_dataset.ans2label:
                        target[self.raw_dataset.ans2label[ans]] = score
                return ques_id, feats, boxes, ques, target
            else:
                return ques_id, feats, boxes, ques
    
    def _uniterBoxes(self, boxes):
        new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
        new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
        new_boxes[:,1] = boxes[:,0]
        new_boxes[:,0] = boxes[:,1]
        new_boxes[:,3] = boxes[:,2]
        new_boxes[:,2] = boxes[:,3]
        new_boxes[:,4] = new_boxes[:,3]-new_boxes[:,1]
        new_boxes[:,5] = new_boxes[:,2]-new_boxes[:,0]
        new_boxes[:,6]=new_boxes[:,4]*new_boxes[:,5]
        return new_boxes


class GQAImageDataset(Dataset):
    """
    Directly load image into memory
    """
    def __init__(self, dataset: GQADataset, img_root="/data8/srip22vg/projects/vg/data/gqa/images"):
        super().__init__()
        self.raw_dataset = dataset
        self.img_root = img_root

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        load_testdev = all(['testdev' in s for s in dataset.splits])
        if load_testdev:
            img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
        else:
            img_data.extend(gqa_buffer_loader.load_data('train', topk))
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item: int):
        datum = self.data[item]

        img = Image.open(os.path.join(self.img_root, f"{datum['img_id']}.jpg")).convert("RGB")

        img_id = datum['img_id']
        ques_id = datum['question_id']
        if 'sent' in datum:
            ques = datum['sent']
        else:
            ques = \
                self.raw_dataset.id2datum[datum['original_question_id']]['sent']

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

        # Create target
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, img, target
        else:
            return ques_id, feats, boxes, ques, img


class GQAViLTDataset(Dataset):
    """
    For ViLT only
    """
    def __init__(self, dataset: GQADataset, img_root="/data8/srip22vg/projects/vg/data/gqa/images"):
        super().__init__()
        self.raw_dataset = dataset
        self.img_root = img_root
        self.transform = keys_to_transforms(["pixelbert_randaug"], size=384)

        # Only kept the data with loaded image features
        self.data = deepcopy(self.raw_dataset.data)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item: int):
        datum = self.data[item]

        img = Image.open(os.path.join(self.img_root, f"{datum['img_id']}.jpg")).convert("RGB")
        img_tensor = self.transform[0](img)

        ques_id = datum['question_id']
        if 'sent' in datum:
            ques = datum['sent']
        else:
            ques = \
                self.raw_dataset.id2datum[datum['original_question_id']]['sent']

        # Create target
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, img_tensor, 0, ques, target
        else:
            return ques_id, img_tensor, 0, ques

def collate_fn(batch):
    batch = list(zip(*batch))
    ques_id = list(batch[0])
    feats = torch.tensor(batch[1], dtype=torch.float)
    boxes = torch.tensor(batch[2], dtype=torch.float)
    ques = list(batch[3])
    img = list(batch[4])
    target = torch.stack(batch[5], 0)
    return ques_id, feats, boxes, ques, img, target

def collate_fn_vilt(batch):
    batch = list(zip(*batch))
    ques_id = list(batch[0])
    boxes = torch.tensor(batch[2], dtype=torch.float)
    ques = list(batch[3])
    target = torch.stack(batch[4], 0)
    # pad images
    img_sizes = [img.shape for img in batch[1]]
    max_height = max([i[1] for i in img_sizes])
    max_width = max([i[2] for i in img_sizes])
    new_images = torch.zeros(len(img_sizes), 3, max_height, max_width)
    for i, orig in enumerate(batch[1]):
        new_images[i, :, :orig.shape[1], :orig.shape[2]] = orig
    new_images = new_images.float()
    return ques_id, new_images, boxes, ques, target


class GQACaptionDataset(Dataset):
    def __init__(self, dataset: GQADataset):
        super().__init__()
        self.raw_dataset = dataset
        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        load_testdev = all(['testdev' in s for s in dataset.splits])
        if load_testdev:
            img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
        else:
            img_data.extend(gqa_buffer_loader.load_data('train', topk))
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

        with open("data/gqa/image2caps.json", 'r') as f:
            self.image2caps = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        if 'sent' in datum:
            ques = datum['sent']
        else:
            ques = \
                self.raw_dataset.id2datum[datum['original_question_id']]['sent']
        
        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        if args.backbone in ['uniter']:
            boxes = self._uniterBoxes(boxes)
        else:
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)
        
        cap = self.image2caps[img_id][0]

        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, cap, ques, target
        else:
            return ques_id, feats, boxes, cap, ques
    
    def _uniterBoxes(self, boxes):
        new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
        new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
        new_boxes[:,1] = boxes[:,0]
        new_boxes[:,0] = boxes[:,1]
        new_boxes[:,3] = boxes[:,2]
        new_boxes[:,2] = boxes[:,3]
        new_boxes[:,4] = new_boxes[:,3]-new_boxes[:,1]
        new_boxes[:,5] = new_boxes[:,2]-new_boxes[:,0]
        new_boxes[:,6]=new_boxes[:,4]*new_boxes[:,5]
        return new_boxes


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }

        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


class GQAOODEvaluator:
    def __init__(self, dataset: GQADataset, tau=0.5):
        self.dataset = dataset
        self.tau = tau
    
    def evaluate(self, quesid2ans: dict):
        """
        minmax: if minmax norm is needed before evaluating ood scores
        """
        from ood_metrics import ood_performance, accfpr_metrics
        from sklearn.metrics import f1_score
        id_acc = 0.
        id_num = 0.
        acc_acc = 0.
        acc_num = 0.
        all_acc = 0.

        ood_preds = []
        ood_targets = []
        clf_targets = []
        clf_preds = []
        for quesid, (ans, score) in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if score > self.tau:
                acc_num += 1
                if ans in label:
                    acc_acc += label[ans]
            if 'UQ' not in label:
                id_num += 1
                if ans in label:
                    id_acc += label[ans]
            if ans in label:
                all_acc += label[ans]
            ood_preds.append(score)
            ood_targets.append(int('UQ' not in label))
            clf_preds.append(ans)
            clf_targets.append(list(label.keys())[0])
        
        ood_preds = np.asarray(ood_preds, dtype=float)
        clf_preds = np.asarray(clf_preds, dtype=str)
        clf_targets = np.asarray(clf_targets, dtype=str)
        results = {}
        results.update(accfpr_metrics(ood_preds, clf_preds, clf_targets, 'UQ', 0.95))
        if acc_num < len(quesid2ans):
            results.update(ood_performance(ood_targets, ood_preds))
            results['accuracy_accept'] = acc_acc / acc_num if acc_num > 0 else 0
            results['accuracy'] = all_acc / len(quesid2ans)
            results['f1'] = f1_score(ood_targets, ood_preds > self.tau)
        results['accuracy_indomain'] = id_acc / id_num
        return results
    
    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }

        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, (ans, conf) in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans,
                    'confidence': round(float(conf), 4)
                })
            json.dump(result, f, indent=4, sort_keys=True)
