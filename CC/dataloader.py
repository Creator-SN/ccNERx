import re
import torch
from CC.utils import DataManager
from torch.autograd import Variable
from ICCSupervised.ICCSupervised import IDataLoader
from torch.utils.data import TensorDataset, DataLoader, Dataset
from CC.loaders.cn_loader import CNDataLoader
from CC.loaders.le_loader import LLoader

class AutoDataLoader(IDataLoader):
    def __init__(self, **args):
        assert "loader_name" in args, "argument loader_name: required loader_name"
        self.loader_name = args['loader_name']
        dataloaders = {
            'cn_loader': CNDataLoader,
            'le_loader': LLoader
        }
        if self.loader_name not in dataloaders:
            raise Exception("Dataloader not found name '{}'.".format(self.loader_name))
        self.loader = dataloaders[self.loader_name](**args)
    
    def __call__(self):
        loader = self.loader
        return loader()