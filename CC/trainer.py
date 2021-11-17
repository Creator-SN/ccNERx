import os
import re
import uuid
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from transformers import BertConfig, BertTokenizer, BertModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from ICCSupervised.ICCSupervised import ITrainer
from CC.dataloader import AutoDataLoader
from CC.analysis import CCAnalysis
from CC.model import CCNERModel
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score


class NERTrainer(ITrainer):

    def __init__(self, **args):
        '''
        Args:
        - num_epochs
        - num_gpus
        - bert_config_file_name
        - pretrained_file_name
        - hidden_dim
        - word_embedding_file: required in `le_loader`
        - word_vocab_file: required in `le_loader`
        - train_file
        - eval_file
        - test_file
        - tag_file
        - bert_vocab_file
        - batch_size
        - eval_batch_size
        - word_tag_split: optional in `cn_loader`, default: ' '
        - pattern: optional in `cn_loader`, default: '， O'
        - output_eval: optional, default: None
        - max_scan: optional in `le_loader`, default: None
        - add_seq_vocab: optional in `le_loader`, default: False
        - max_seq_length: optional in `le_loader`, default: 256
        - max_word_num: optional in `le_loader`, default: 5
        - default_tag: optional in `le_loader`, default: "O"
        - model_name: optional, default: "LEBert"
        - loader_name: optional, default: "le_loader"
        - task_name: optional, default: None
        '''
        assert "num_epochs" in args, "argument num_epochs: required embeding file path"
        assert "num_gpus" in args, "argument num_gpus: required embeding file path"
        assert "hidden_dim" in args, "argument hidden_dim: required embeding file path"
        self.model_name: str = 'LEBert'
        if "model_name" in args:
            self.model_name = args["model_name"]
        self.loader_name = 'le_loader'
        if "loader_name" in args:
            self.loader_name = args["loader_name"]

        self.eval_data = args['output_eval']
        self.num_epochs = args['num_epochs']
        self.num_gpus = args['num_gpus']
        self.output_eval = args['output_eval']
        self.dataloader_init(**args)
        self.model_init(**args)
        self.task_name = args['task_name']

    def model_init(self, **args):
        model_args = {
            'model_name': self.model_name,
            'bert_config_file_name': args['bert_config_file_name'],
            'tagset_size': self.tag_size,
            'hidden_dim': args['hidden_dim']
        }
        if 'word_embedding_file' in args:
            model_args['pretrained_embeddings'] = self.vocab_embedding
        if 'tag_embedding_file' in args:
            model_args['label_embeddings'] = self.label_embedding
        if 'pretrained_file_name' in args:
            model_args['pretrained_file_name'] = args['pretrained_file_name']

        self.bert_ner = CCNERModel(**model_args)
        self.model, self.birnncrf = self.bert_ner()

    def dataloader_init(self, **args):
        self.dataloader = AutoDataLoader(**args)
        result = self.dataloader()
        self.train_data = result['train_set']
        self.train_iter = result['train_iter']
        if self.loader_name == 'le_loader':
            self.vocab_embedding = result['vocab_embedding']
            self.embedding_dim = result['embedding_dim']
            self.tag_vocab = result['tag_vocab']
            self.tag_size = self.tag_vocab.__len__()
            self.analysis = CCAnalysis(
                self.tag_vocab.token2id, self.tag_vocab.id2token)
        if self.loader_name == 'cn_loader':
            self.dm = result['dm']
            self.tag_size = len(self.dm.tag_to_idx)
            self.analysis = CCAnalysis(self.dm.tagToIdx, self.dm.idxToTag)
        
        if self.loader_name == 'labelle_loader':
            self.vocab_embedding = result['vocab_embedding']
            self.embedding_dim = result['embedding_dim']
            self.label_embedding = result['label_embedding']
            self.label_embedding_dim = result['label_embedding_dim']
            self.tag_vocab = result['tag_vocab']
            self.tag_size = self.tag_vocab.__len__()
            self.analysis = CCAnalysis(
                self.tag_vocab.token2id, self.tag_vocab.id2token)

        if self.output_eval is not None:
            self.eval_set = result['eval_set']
            self.eval_iter = result['eval_iter']

    def train(self, resume_path=False, resume_step=False, lr1=2e-5, lr2=1e-3):
        alpha = 1e-10

        optimizer = optim.AdamW([
            {'params': self.model.parameters(), 'lr': lr1},
            {'params': self.birnncrf.parameters(), 'lr': lr2}
        ], lr=1e-5, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 0:
            self.model = nn.DataParallel(self.model, device_ids=self.num_gpus)
            self.birnncrf.cuda()

        self.model.to(device)
        self.birnncrf.to(device)

        if not resume_path == False:
            print('Accessing Resume PATH: {} step: {} ...\n'.format(
                resume_path, resume_step))
            bert_model_dict = torch.load(os.path.join(
                resume_path, self.model_name, '{}_{}.pth'.format(self.model_name, resume_step))).module.state_dict()
            self.model.module.load_state_dict(bert_model_dict)
            self.birnncrf = torch.load(os.path.join(
                resume_path, 'lstm_crf', 'lstm_crf_{}.pth'.format(resume_step)))
            self.model.to(device)
            self.birnncrf.to(device)

        current_uid = str(uuid.uuid1()).split('-')[0]

        train_step = resume_step if resume_step != False else 0
        for epoch in range(self.num_epochs):
            train_count = 0
            train_loss = 0
            train_iter = tqdm(self.train_iter)
            self.model.train()
            self.birnncrf.train()

            all_acc_list = []
            all_p_list = []
            all_r_list = []
            all_f1_list = []
            for it in train_iter:
                pred_labels_list = []
                true_labels_list = []

                train_step += 1

                for key in it.keys():
                    # temp_list = it[key].tolist()
                    # temp = []
                    # for item in temp_list:
                    #     temp.append(item[:])
                    #     temp.append(item[:])
                    # it[key] = temp
                    # it[key] = self.cuda(torch.tensor(it[key]))
                    it[key] = self.cuda(it[key])

                outputs = self.model(**it)
                hidden_states = outputs['mix_output']
                loss = self.birnncrf.loss(
                    hidden_states, it['input_ids'].gt(0), it['labels'])
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
                self.birnncrf.zero_grad()

                train_loss += loss.data.item()
                train_count += 1

                pred = self.birnncrf(hidden_states, it['input_ids'].gt(0))[1]

                for item_index in range(it["input_ids"].shape[0]):
                    # remove [PAD] length
                    real_length = 0
                    for idx in range(it['input_ids'][item_index].shape[0]-1, -1, -1):
                        if it["input_ids"][item_index][idx] > 0:
                            real_length = idx+1
                            break
                    # remove [SEP] and [CLS]
                    pred_labels = pred[item_index][1:real_length-1]
                    true_labels = it['labels'][item_index].tolist()[
                        1:real_length-1]

                    pred_labels = [label.replace(
                        "M-", "I-") for label in self.analysis.idx2tag(pred_labels)]
                    true_labels = [label.replace(
                        "M-", "I-") for label in self.analysis.idx2tag(true_labels)]
                    
                    pred_labels_list.append(pred_labels)
                    true_labels_list.append(true_labels)

                acc = accuracy_score(true_labels_list, pred_labels_list)
                p = precision_score(
                    true_labels_list, pred_labels_list)
                r = recall_score(true_labels_list, pred_labels_list)
                f1 = f1_score(true_labels_list, pred_labels_list)

                all_acc_list.append(acc)
                all_p_list.append(p)
                all_r_list.append(r)
                all_f1_list.append(f1)

                train_acc = np.mean(all_acc_list)
                train_precision = np.mean(all_p_list)
                train_recall = np.mean(all_r_list)
                F1 = np.mean(all_f1_list)

                train_iter.set_description(
                    'Epoch: {}/{} Train'.format(epoch + 1, self.num_epochs))
                train_iter.set_postfix(train_loss=train_loss / train_count, train_acc=train_acc, train_precision=train_precision,
                                       train_recall=train_recall, F1=F1)
                #    =(2 * train_acc * train_recall) / (train_acc + train_recall + alpha))
                self.analysis.append_train_record({
                    'loss': loss.data.item(),
                    # 'f1': (2 * train_acc * train_recall) / (train_acc + train_recall + alpha),
                    'f1': F1,
                    'acc': train_precision,
                    'recall': train_recall
                })

            model_uid = self.save_model(train_step)
            if self.eval_data:
                self.eval()

            self.analysis.save_ner_record(
                uid=current_uid if self.task_name is None else self.task_name)
            yield (epoch + 1, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, model_uid)

    def eval(self):
        alpha = 1e-10

        test_count = 0
        eval_loss = 0
        test_iter = tqdm(self.eval_iter)
        self.model.eval()
        self.birnncrf.eval()
        with torch.no_grad():
            all_acc_list = []
            all_p_list = []
            all_r_list = []
            all_f1_list = []
            for it in test_iter:
                pred_labels_list = []
                true_labels_list = []

                for key in it.keys():
                    it[key] = self.cuda(it[key])

                outputs = self.model(**it)
                hidden_states = outputs['mix_output']
                loss = self.birnncrf.loss(
                    hidden_states, it['input_ids'].gt(0), it['labels'])
                loss = loss.mean()

                eval_loss += loss.data.item()
                test_count += 1

                pred = self.birnncrf(hidden_states, it['input_ids'].gt(0))[1]

                for item_index in range(it["input_ids"].shape[0]):
                    # remove [PAD] length
                    real_length = 0
                    for idx in range(it['input_ids'][item_index].shape[0]-1, -1, -1):
                        if it["input_ids"][item_index][idx] > 0:
                            real_length = idx+1
                            break
                    # remove [SEP] and [CLS]
                    pred_labels = pred[item_index][1:real_length-1]
                    true_labels = it['labels'][item_index].tolist()[
                        1:real_length-1]
                    pred_labels = [label.replace(
                        "M-", "I-") for label in self.analysis.idx2tag(pred_labels)]
                    true_labels = [label.replace(
                        "M-", "I-") for label in self.analysis.idx2tag(true_labels)]

                    pred_labels_list.append(pred_labels)
                    true_labels_list.append(true_labels)

                acc = accuracy_score(true_labels_list, pred_labels_list)
                p = precision_score(
                    true_labels_list, pred_labels_list)
                r = recall_score(true_labels_list, pred_labels_list)
                f1 = f1_score(true_labels_list, pred_labels_list)

                all_acc_list.append(acc)
                all_p_list.append(p)
                all_r_list.append(r)
                all_f1_list.append(f1)

                test_acc = np.mean(all_acc_list)
                test_precision = np.mean(all_p_list)
                test_recall = np.mean(all_r_list)
                F1 = np.mean(all_f1_list)

                # t1, t2 = self.analysis.getPrecision(it['labels'], pred)
                # test_pred_num += t1
                # test_correct_num += t2
                # test_gold_num += self.analysis.getRecall(it['labels'])

                # test_acc = test_correct_num / test_pred_num if test_pred_num != 0 else 0
                # test_recall = test_correct_num / test_gold_num if test_gold_num != 0 else 0

                test_iter.set_description('Eval Result')
                test_iter.set_postfix(
                    eval_loss=eval_loss / test_count, eval_acc=test_acc, eval_precision=test_precision, eval_recall=test_recall, F1=F1)
                #   F1=(2 * test_acc * test_recall) / (test_acc + test_recall + alpha))
            self.analysis.append_eval_record({
                'loss': loss.data.item(),
                # 'f1': (2 * test_acc * test_recall) / (test_acc + test_recall + alpha),
                'f1': F1,
                'acc': test_precision,
                'recall': test_recall
            })

    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        if not os.path.exists('./save_model/{}/{}'.format(dir, self.model_name)):
            os.makedirs('./save_model/{}/{}'.format(dir, self.model_name))
        if not os.path.exists('./save_model/{}/lstm_crf'.format(dir)):
            os.makedirs('./save_model/{}/lstm_crf'.format(dir))
        torch.save(
            self.model, './save_model/{}/{}/{}_{}.pth'.format(dir, self.model_name, self.model_name, current_step))
        torch.save(self.birnncrf,
                   './save_model/{}/lstm_crf/lstm_crf_{}.pth'.format(dir, current_step))
        self.analysis.append_model_record(current_step)
        return current_step

    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available():
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available():
                return inputX.cuda()
            return inputX

    def __call__(self, **arg):
        for r in self.train(**arg):
            yield r
