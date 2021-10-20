# %%
from CC.loaders import LLoader

args = {
    "batch_size": 4,
    "eval_batch_size": 16,
    "test_file": "./data/test.json",
    "eval_file": "./data/dev.json",
    "train_file": "./data/train.json",
    "tag_file": "./data/labels.txt",
    "word_embedding_file": "./data/tencent/word_embedding.txt",
    "word_vocab_file": "./data/tencent/tencent_vocab.txt",
    "bert_vocab_file": "./data/bert/chinese_wwm_ext/vocab.txt",
    "default_tag": "O",
    "max_scan_num": 1500000
}

LLoader(**args).myData[0]

# %%
import json
from CC.trainer import NERTrainer
from CC.predicter import NERPredict
#%%
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    'hidden_dim': 150,
    'max_seq_length': 50,
    'word_tag_split': ' ',
    'pattern': '， O',
    'train_file': './data/news/train.txt',
    'eval_file': './data/news/dev.txt',
    'test_file': './data/news/test.txt',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/news_tags_list.txt',
    'output_eval': True,
    'loader_name': 'cn_loader',
    'batch_size': 32,
    'eval_batch_size': 64,
    'model_name': 'Bert',
    'task_name': 'news'
}

# %%

trainer = NERTrainer(**args)

for i in trainer():
    a = i

# %%
from CC.loaders.utils import *
class foo():
    def __init__(self,**kwargs):
        KwargsParser(True) \
        .add_argument("num_epochs", int, 30) \
        .add_argument("num_gpus", list) \
        .parse(self,**kwargs)

a = foo(**args)
a.num_gpus
    
# %%
