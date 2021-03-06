# %%
import json
from CC.predicter import NERPredict
from CC.trainer import NERTrainer

# %%
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './save_pretrained/Weibo_CNX_Pretrained/Bert_2760/pytorch_model.bin',
    'hidden_dim': 300,
    'max_seq_length': 150,
    'max_scan_num': 1000000,
    'train_file': './data/weibonew/train_origin.json',
    'eval_file': './data/weibo/dev.json',
    'test_file': './data/weibo/test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/weibo/labels.txt',
    'output_eval': True,
    'loader_name': 'le_loader',
    "word_embedding_file":"./data/tencent/word_embedding.txt",
    "word_vocab_file":"./data/tencent/tencent_vocab.txt",
    "default_tag":"O",
    'batch_size': 8,
    'eval_batch_size': 64,
    'do_shuffle': True,
    'model_name': 'LEBert',
    'task_name': 'weibo_new_02'
}

# %%
predict = NERPredict(**args)
predict(["坐落于福州的福州大学ACM研究生团队, 在帅气幽默的傅仰耿老师带领下, 正在紧张刺激的开发一套全新的神秘系统。"])

# %%
trainer = NERTrainer(**args)

for i in trainer():
    a = i


# %%
from CC.loaders import *
import pickle
from tqdm import *
from CC.loaders.utils import *
import json
from CC.pre_trained import NERPreTrainer

# %%
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    'max_seq_length': 512,
    'max_scan_num': 1500000,
    'train_file': './data/SuperNER/pre_train.json',
    'eval_file': './data/SuperNER/pre_dev.json',
    'test_file': './data/SuperNER/pre_test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/SuperNER/tags_list.txt',
    'loader_name': 'lex_loader',
    "word_embedding_file": "./data/tencent/word_embedding.txt",
    "word_vocab_file": "./data/tencent/tencent_vocab.txt",
    "word_vocab_file_with_tag": "./data/tencent/tencent_vocab_with_tag.json",
    "default_tag": "O",
    'batch_size': 32,
    'eval_batch_size': 64,
    'do_shuffle': True,
    'task_name': 'Pre_trained',
    "use_gpu": True,
    "debug": True,
    "tag_rules": {
        "ORG": "组织",
        "LOC": "地点",
        "PER": "人",
        "Time": "时间",
        "Thing": "物品",
        "Metric": "测量单位",
        "Abstract": "作品",
        "Physical": "实体",
        "Term": "术语",
        "company": "企业",
        "name": "名字",
        "game": "游戏",
        "movie": "电影",
        "position": "职位",
        "address": "地址",
        "government": "政府",
        "scene": "景点",
        "book": "书名"
    }
}

pre_trainer = NERPreTrainer(**args)

for i in pre_trainer():
    a = i

# %%
# This is a pre-trained template with lex_loader

# %%
from CC.loaders import *
import pickle
from tqdm import *
from CC.loaders.utils import *
import json
from CC.pre_trained import NERPreTrainer

# %%
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    'max_seq_length': 512,
    'max_scan_num': 1000000,
    'train_file': './data/weibonew/train.json',
    'eval_file': './data/weibo/dev.json',
    'test_file': './data/weibo/test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/weibo/pretrained_labels.txt',
    'loader_name': 'lex_loader',
    "word_embedding_file": "./data/tencent/word_embedding.txt",
    "word_vocab_file": "./data/tencent/tencent_vocab.txt",
    "word_vocab_file_with_tag": "./data/tencent/tencent_vocab_with_tag.json",
    "default_tag": "O",
    'batch_size': 32,
    'eval_batch_size': 64,
    'do_shuffle': True,
    'task_name': 'Weibo_x20_Pretrained',
    "use_gpu": True,
    "debug": True,
    "tag_rules": {
        "PER.NOM": "人的象征",
        "LOC.NAM": "地点",
        "PER.NAM": "人",
        "GPE.NAM": "政治实体",
        "ORG.NAM": "组织",
        "ORG.NOM": "组织的象征",
        "LOC.NOM": "地点的象征",
        "GPE.NOM": "政治实体的象征",
        "ORG": "组织",
        "LOC": "地点",
        "PER": "人",
        "Time": "时间",
        "Thing": "物品",
        "Metric": "测量单位",
        "Abstract": "作品",
        "Physical": "实体",
        "Term": "术语",
        "company": "企业",
        "name": "名字",
        "game": "游戏",
        "movie": "电影",
        "position": "职位",
        "address": "地址",
        "government": "政府",
        "scene": "景点",
        "book": "书名"
    }
}

pre_trainer = NERPreTrainer(**args)

for i in pre_trainer():
    a = i

# %%
# This is a pre-trained template with cnx_loader

# %%
from CC.loaders import *
import pickle
from tqdm import *
from CC.loaders.utils import *
import json
from CC.pre_trained import NERPreTrainer

# %%
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    'max_seq_length': 512,
    'max_scan_num': 1000000,
    'train_file': './data/weibonew/train.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/weibo/pretrained_labels.txt',
    'loader_name': 'cnx_loader',
    'batch_size': 32,
    'eval_batch_size': 64,
    'do_shuffle': True,
    'use_json': True,
    'model_name': 'Bert',
    'task_name': 'Weibo_CNX_Pretrained',
    "use_gpu": True,
    "debug": True,
    "tag_rules": {
        "PER.NOM": "人的象征",
        "LOC.NAM": "地点",
        "PER.NAM": "人",
        "GPE.NAM": "政治实体",
        "ORG.NAM": "组织",
        "ORG.NOM": "组织的象征",
        "LOC.NOM": "地点的象征",
        "GPE.NOM": "政治实体的象征",
        "ORG": "组织",
        "LOC": "地点",
        "PER": "人",
        "Time": "时间",
        "Thing": "物品",
        "Metric": "测量单位",
        "Abstract": "作品",
        "Physical": "实体",
        "Term": "术语",
        "company": "企业",
        "name": "名字",
        "game": "游戏",
        "movie": "电影",
        "position": "职位",
        "address": "地址",
        "government": "政府",
        "scene": "景点",
        "book": "书名"
    }
}

pre_trainer = NERPreTrainer(**args)

for i in pre_trainer():
    a = i

# %%