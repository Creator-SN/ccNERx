import os
import uuid
from ICCSupervised.ICCSupervised import IAnalysis

class CCAnalysis(IAnalysis):

    def __init__(self, tag_vocab):
        self.train_record = {
            'loss': [],
            'f1': [],
            'acc': [],
            'recall': []
        }

        self.eval_record = {
            'loss': [],
            'f1': [],
            'acc': [],
            'recall': []
        }

        self.model_record = []

        self.tag_vocab = tag_vocab
    
    def append_train_record(self, train_record_item):
        for key in train_record_item:
            self.train_record[key].append(train_record_item[key])
    
    def append_eval_record(self, eval_record_item):
        for key in eval_record_item:
            self.eval_record[key].append(eval_record_item[key])
    
    def append_model_record(self, uid):
        self.model_record.append(uid)
    
    def getPrecision(self, tags, pred):
        pred_num = 0
        correct_num = 0
        for i, item in enumerate(pred):
            t_tags = tags[i].tolist()
            insider = False
            eq_tag = False
            for j, tag in enumerate(item):
                if tag != self.tag_vocab.token2id('O'):
                    if insider == False:
                        pred_num += 1
                        insider = True
                        eq_tag = True
                    if tag != t_tags[j]:
                        eq_tag = False
                else:
                    insider = False
                    if eq_tag == True:
                        correct_num += 1
                    eq_tag = False
        return pred_num, correct_num


    def getRecall(self, tags):
        gold_num = 0
        for i, item in enumerate(tags):
            t_tags = item.tolist()
            for tag in t_tags:
                if self.tag_vocab.id2token(tag).find('B-') > -1:
                    gold_num += 1
                elif self.tag_vocab.id2token(tag).find('S-') > -1:
                    gold_num += 1
        return gold_num
    
    def save_csv(self, uid):
        if not os.path.exists('./data_record'):
            os.makedirs('./data_record')
        with open('./data_record/{}_train.csv'.format(uid), encoding='utf-8', mode='w+') as f:
            result = 'F1,Acc,Recall,loss\n'
            for i, item in enumerate(self.train_record['loss']):
                result += '{},{},{},{}\n'.format(self.train_record['f1'][i], self.train_record['acc'][i], self.train_record['recall'][i], self.train_record['loss'][i])
            f.write(result)
        
        with open('./data_record/{}_eval.csv'.format(uid), encoding='utf-8', mode='w+') as f:
            result = 'F1,Acc,Recall,loss\n'
            for i, item in enumerate(self.eval_record['loss']):
                result += '{},{},{},{}\n'.format(self.eval_record['f1'][i], self.eval_record['acc'][i], self.eval_record['recall'][i], self.eval_record['loss'][i])
            f.write(result)
        
        with open('./data_record/{}_model.csv'.format(uid), encoding='utf-8', mode='w+') as f:
            result = 'model_uid\n'
            for i, item in enumerate(self.model_record):
                result += '{}\n'.format(item)
            f.write(result)
        return uid