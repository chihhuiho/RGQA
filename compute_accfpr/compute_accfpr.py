"""
The file for training gqa-uq. Mostly copied from gqa.py.
"""

import os
import json
import pickle
import collections
import numpy as np
from sklearn.metrics import f1_score
from ood_metrics import ood_performance, accfpr_metrics

class GQAOODEvaluator:
    def __init__(self, qid2label, tau=0.5):
        self.qid2label = qid2label
        self.tau = tau
    
    def evaluate(self, quesid2ans: dict):
        """
        minmax: if minmax norm is needed before evaluating ood scores
        """
        id_acc = 0.
        id_num = 0.
        acc_acc = 0.
        acc_num = 0.
        all_acc = 0.

        ood_preds = []
        ood_targets = []
        clf_targets = []
        clf_preds = []
        for datum in quesid2ans:
            # {'confidence': 0.9701, 'prediction': 'color', 'questionId': '20226371'}
            score = datum['confidence']
            ans = datum['prediction']
            quesid = datum['questionId']

            label = self.qid2label[quesid]
            # for AQ is the VQA answer class {'ottoman': 1.0}
            # for UQ label is {'UQ': 1.0}
 
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
    
   
def ood_evaluate(qid2label, quesid2ans):
    evaluator = GQAOODEvaluator(qid2label)
    results = evaluator.evaluate(quesid2ans)
    return results



if __name__ == "__main__":
    f = open("example.json")
    rgqa_dataset = json.load(f) # list of {'img_id': 'n68769', 'label': {'color': 1.0}, 'question_id': '20226371', 'sent': 'What do both the ceiling and the table have in common?'}
    f.close()

    f = open("example_predict.json")
    quesid2ans = json.load(f) # list of {'confidence': 0.9701, 'prediction': 'color', 'questionId': '20226371'}
    f.close()
   

    qid2label = {}
    for q in rgqa_dataset:
        qid2label[q['question_id']] = q['label']
 
    result = ood_evaluate(qid2label, quesid2ans)
    print(result)
