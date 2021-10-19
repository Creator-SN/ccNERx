import re
import torch
from CC.utils import DataManager
from torch.autograd import Variable
from ICCSupervised.ICCSupervised import IDataLoader
from torch.utils.data import TensorDataset, DataLoader, Dataset

class CNDataLoader(IDataLoader):
    def __init__(self, train_file_name, vocab_file_name, tags_file_name, word_tag_split=' ', eval_file_name=None, pattern='， O', padding_length=50, batch_size=32, eval_batch_size=None):
        if eval_file_name is not None:
            self.output_eval = True
        else:
            self.output_eval = False
        self.read_data_set(train_file_name, eval_file_name, word_tag_split, pattern)
        self.data_manager_init(vocab_file_name, tags_file_name)
        self.verify_data()
        self.process_data(padding_length, batch_size, eval_batch_size)
    
    '''
    全局数据控制器
    vocab_file_name: 单词表文件名
    tags_file_name: 标签表文件名
    '''
    def data_manager_init(self, vocab_file_name, tags_file_name):
        tags_list = BERTDataManager.ReadTagsList(tags_file_name)
        tags_list = [tags_list]
        self.dm = BERTDataManager(tags_list=tags_list,
                     vocab_file_name=vocab_file_name)
    
    '''
    读取数据
    '''
    def read_data_set(self, train_file_name, eval_file_name=None, word_tag_split=' ', pattern='， O'):
        self.train_set, self.train_tags = BERTDataManager.ReadDataExtremely(train_file_name, word_tag_split, pattern)
        if eval_file_name is not None:
            self.eval_set, self.eval_tags = BERTDataManager.ReadDataExtremely(eval_file_name, word_tag_split, pattern)
    
    '''
    验证数据
    '''
    def verify_data(self):
        if len(self.train_set) != len(self.train_tags):
            raise Exception("The number of train_sentence is not equal to train_tags.")
        for i, item in enumerate(self.train_set):
            if len(self.train_set[i]) != len(self.train_tags[i]):
                raise Exception("The length of sentence at index: {} is not equal to the length of corresponding tags.".format(i))
            for tag in self.train_tags[i]:
                if tag not in self.dm.tag_to_idx:
                    raise Exception("The tag of '{}' is not exists in tags_list.".format(tag))

    '''
    处理数据
    '''
    def process_data(self, padding_length, batch_size, eval_batch_size):
        eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
        self.myData = CCNERDataset(self.train_set, self.train_tags, self.dm, padding_length)
        self.dataiter = DataLoader(self.myData, batch_size=batch_size)
        if self.output_eval:
            self.myData_eval = CCNERDataset(self.eval_set, self.eval_tags, self.dm, padding_length)
            self.dataiter_eval = DataLoader(self.myData_eval, batch_size=eval_batch_size)
            
    
    def __call__(self):
        if self.output_eval:
            return [(self.myData, self.dataiter, self.dm), (self.myData_eval, self.dataiter_eval)]
        return [(self.myData, self.dataiter, self.dm)]

class CCNERDataset(Dataset):
    def __init__(self, sentences_list, tags_list, data_manager, padding_length=100):
        self.sentences, self.tags_list = sentences_list, tags_list
        self.data_manager = data_manager
        self.padding_length = padding_length
    
    def __getitem__(self, idx):
        sentence, tags = self.data_manager.encode(
            self.sentences[idx], self.tags_list[idx], padding_length=self.padding_length - 1)
        sentence = [101] + sentence
        tags = [self.data_manager.tag_to_idx['[CLS]']] + tags
        for i, _ in enumerate(sentence):
            if _ == 0:
                sentence[i] = 102
                tags[i] = self.data_manager.tag_to_idx['[SEP]']
                break
        sentence = torch.tensor(sentence)
        tags = torch.tensor(tags)
        return sentence, tags
    
    def __len__ (self):
        return len(self.sentences)

class BERTDataManager(DataManager):
    @staticmethod
    def TagIdxInit(tags_list, pad_tag='[PAD]'):
        count = 1
        tag_to_idx = {}
        idx_to_tag = {}
        tag_to_idx[pad_tag] = 0
        idx_to_tag[0] = pad_tag
        for item in tags_list:
            for tag in item:
                if tag not in tag_to_idx:
                    tag_to_idx[tag] = count
                    idx_to_tag[count] = tag
                    count += 1
        tag_to_idx['[CLS]'] = count
        idx_to_tag[count] = '[CLS]'
        count += 1
        tag_to_idx['[SEP]'] = count
        idx_to_tag[count] = '[SEP]'
        return tag_to_idx, idx_to_tag