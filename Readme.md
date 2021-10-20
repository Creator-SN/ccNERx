# ccNERx

- ICCSupervised
- CC
    - loaders
        - utils
            - embedding.py
            - file_util.py
            - lexicon_factory.py
            - lexicon_tree.py
            - vocab.py
        - cn_loader.py
        - le_loader.py
    - analysis.py
    - birnncrf.py
    - crf.py
    - dataloader.py
    - LEBert.py
    - model.py
    - predicter.py
    - trainer.py
    - utils.py

## Requirements

**environment**

```bash
transformers==4.5.1
torch
```

**LLoader requirements**

1. Word vocab

[Word vocab](https://drive.google.com/file/d/1UmtbCSPVrXBX_y4KcovCknJFu9bXXp12/view?usp=sharing)

2. Word embedding

[Chinese word embedding](https://ai.tencent.com/ailab/nlp/en/data/Tencent_AILab_ChineseEmbedding.tar.gz)

## Demo

**Trainer LELoader**

```python
from CC.predicter import NERPredict
from CC.trainer import NERTrainer

# %%
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    'hidden_dim': 300,
    'max_seq_length': 150,
    'max_scan_num': 1500000,
    'train_file': './data/weibo/train.json',
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
    'model_name': 'LEBert',
    'task_name': 'weibo'
}
trainer = NERTrainer(**args)

for i in trainer():
    a = i
```

**Trainer CNLoader**

```python
from CC.trainer import NERTrainer
from CC.predicter import NERPredict

# %%
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
trainer = NERTrainer(**args)

for i in trainer():
    a = i
```

**Predictor**

```python
# %%
from predicter import NERPredict

# %%
predict = NERPredict(True,
                     bert_config_file_name='./model/chinese_wwm_ext/bert_config.json',
                     vocab_file_name='./model/chinese_wwm_ext/vocab.txt',
                     tags_file_name='./data/news_tags_list.txt',
                     bert_model_path='./save_model/bert/cbef37de_bert.pth',
                     lstm_crf_model_path='./save_model/lstm_crf/cbef37de_lstm_crf.pth',
                     hidden_dim=150)

# %%
print(predict(["坐落于福州的福州大学ACM研究生团队, 在帅气幽默的傅仰耿老师带领下, 正在紧张刺激的开发一套全新的神秘系统。","在福州大学的后山, 驻扎着福大后山协会, 会长是陈学勤同志。"])[2:])


# %%
labels, text = predict(["福建省能源集团有限责任公司（以下简称集团）成立于2009年12月,是由福建省煤炭工业（集团）有限责任公司和福建省建材（控股）有限责任公司整合重组而成，系福建省属国有企业，2015年7月起并表福建石油化工集团有限责任公司。集团拥有全资及控股并表企业176家，在职员工2万余人，其中福能股份、福建水泥在主板上市，福能租赁、福能期货在新三板挂牌。集团注册资本金100亿元，资信等级连续多年保持AAA级别。集团连年列入中国企业500强。"])[2:]

# %%
labels, text = predict(["欢迎福能集团黄守清院长莅临福州大学ACM团队指导工作！"])[2:]

# %%
for idx, label in enumerate(labels):
    t = text[idx]
    for j, item in enumerate(label):
        print('{}\t{}'.format(t[j], item))

# %%
```