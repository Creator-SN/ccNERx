import re
import torch
from ner.utils import DataManager
from torch.autograd import Variable
from ICCSupervised.ICCSupervised import IDataLoader
from torch.utils.data import TensorDataset, DataLoader, Dataset
from ner.loaders.cn_loader import CNDataLoader

class AutoDataLoader(IDataLoader):
    def __init__(self, train_file_name, vocab_file_name, tags_file_name, word_tag_split=' ', eval_file_name=None, pattern='， O', padding_length=50, batch_size=32, eval_batch_size=None, loader_name='cn_loader'):
        self.loader_name = loader_name
        if eval_file_name != None:
            self.output_eval = True
        else:
            self.output_eval = False
        dataloaders = {
            'cn_loader': CNDataLoader
        }
        if self.loader_name not in dataloaders:
            raise Exception("Dataloader not found name '{}'.".format(self.loader_name))
        self.loader = dataloaders[self.loader_name](train_file_name, vocab_file_name, tags_file_name, word_tag_split, eval_file_name, pattern, padding_length, batch_size, eval_batch_size)
    
    def __call__(self):
        loader = self.loader
        if self.output_eval:
            return [(loader.myData, loader.dataiter, loader.dm), (loader.myData_eval, loader.dataiter_eval)]
        return [(loader.myData, loader.dataiter, loader.dm)]