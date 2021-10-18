import re
import torch
from ner.utils import DataManager
from torch.autograd import Variable
from ICCSupervised.ICCSupervised import IDataLoader
from torch.utils.data import TensorDataset, DataLoader, Dataset
from ner.loaders.cn_loader import CNDataLoader
from ner.loaders.l_loader import LLoader

class AutoDataLoader(IDataLoader):
    def __init__(self, **args):
        assert "word_embedding_file" in args, "argument word_embedding_file: required embeding file path"
        assert "word_vocab_file" in args, "argument word_vocab_file: required word vocab file to build lexicon tree"
        assert "train_file" in args, "argument train_file: required train file path"
        # assert "eval_file" in args, "argument eval_file: required eval file path"
        assert "test_file" in args, "argument test_file: required test file path"
        assert "tag_file" in args, "argument tag_file: required label file path"
        assert "loader_name" in args, "argument loader_name: required loader_name file path"
        self.loader_name = args['loader_name']
        if 'eval_file' not in args:
            self.output_eval = True
        else:
            self.output_eval = False
        dataloaders = {
            'cn_loader': CNDataLoader,
            'l_loader': LLoader
        }
        if self.loader_name not in dataloaders:
            raise Exception("Dataloader not found name '{}'.".format(self.loader_name))
        self.loader = dataloaders[self.loader_name](**args)
    
    def __call__(self):
        loader = self.loader
        return loader()