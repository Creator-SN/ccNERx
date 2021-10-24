# %%
from tqdm import *
from CC.loaders.utils import *
import json
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
    'max_scan_num': 1000000,
    'train_file': './data/weibo/train.json',
    'eval_file': './data/weibo/dev.json',
    'test_file': './data/weibo/test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    "bert_model_file": "./save_model/weibo/LEBert/LEBert_845.pth",
    'tag_file': './data/weibo/labels.txt',
    'loader_name': 'le_loader',
    "word_embedding_file": "./data/tencent/word_embedding.txt",
    "word_vocab_file": "./data/tencent/tencent_vocab.txt",
    "default_tag": "O",
    'batch_size': 8,
    'eval_batch_size': 64,
    'model_name': 'LEBert',
    'task_name': 'weibo',
    'lstm_crf_model_file': './save_model/weibo/lstm_crf/lstm_crf_845.pth',
    "use_gpu": True,
}

# %%
predict = NERPredict(**args)

# %%

with open('./data/weibo/dev.json') as f:
    ori_list = f.read().split('\n')
ori_list = ori_list[:-1]

batch_size = 50
texts = []
result = []
for idx, item in enumerate(ori_list):
    item = json.loads(item)
    item = item['text']
    texts.append(''.join(item))
    if idx != 0 and idx % batch_size == 0:
        result += predict(texts[idx - batch_size:idx])

if idx % batch_size != 0:
    result += predict(texts[idx - (idx % batch_size):])

with open('./weibo_predict.json', mode='w+') as f:
    f.write("")
with open('./weibo_predict.json', mode='a+') as f:
    for line in result:
        f.write('{}\n'.format(json.dumps({
            'text': line[0],
            'label': line[1]
        }, ensure_ascii=False)))

# %%
