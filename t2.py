# %%
from CC.loaders import *
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    'hidden_dim': 300,
    'max_seq_length': 150,
    'max_scan_num': 100,
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
    'batch_size': 8,
    'eval_batch_size': 64,
    'model_name': 'LEXBert',
    'task_name': 'Loader_test',
    "use_gpu": True,
    "debug": True,
    "tag_rules": {
        "ORG": "组织",
        "LOC": "地点",
        "PER": "人",
        "Time": "时间",
        "Thing": "物品",
        "Metric": "测量单位",
        "Abstract": "抽象",
        "Physical": "身体部位",
        "Term": "学期",
        "company": "企业",
        "name": "名字",
        "game": "游戏",
        "movie": "电影",
        "position": "职位",
        "address": "地址",
        "government": "政府",
        "scene": "景点",
        "book": "书名",
    }
}
loader = LXLoader(**args)

# %%
loader.myData[0]["input_ids"]
loader.tokenizer.decode(loader.myData[0]["input_ids"])

# %%
loader.tag_vocab.id2token(loader.myData[0]["labels"].tolist())

# %%
loader.tokenizer.decode(loader.myData[0]["origin_labels"])

# %%
predict = NERPredict(**args)

# %%
predict(["哦啊", "词语", "福州", "测试", "你好", "助手", "帮忙",
        "团结", "友善", "民主", "善良", "小伙子", "新浪财经", "微博", "博客"])
# %%
filename = "./data/tencent/tencent_vocab.txt"
totals = FileUtil.count_lines(filename)
iter = tqdm(FileUtil.line_iter(filename), total=totals)
savefile = "./data/tencent/tencent_vocab_with_tag.json"
with open(savefile, "w", encoding="utf-8") as f:
    words = []
    for index, line in enumerate(iter):
        word = line.strip()
        words.append(word)
        if index % 2048 == 2047:
            preds = predict(words)
            buffer = "\n".join(
                [json.dumps(pred, ensure_ascii=False) for pred in preds])
            # print(buffer)
            f.write(f'{buffer}\n')
            words = []
    if len(words) > 0:
        preds = predict(words)
        buffer = "\n".join([json.dumps(pred, ensure_ascii=False)
                           for pred in preds])
        f.write(f'{buffer}\n')
        words = []


# %%
loader = LabelLoader(**{
    "auto_loader": False,
    "debug": True,
    "file_name": "data/weibo/train.json",
    "random_rate": 0.2,
    "expansion_rate": 50
}).read_data_set("data/weibonew/train_origin.json", 1.0) \
    .process_data(20) \
    .to_file("./data/weibonew/train_20_2.json")

# .to_file("./data/weibonew/train_origin.json") \

# %%
print(FileUtil.count_lines("./data/weibonew/train.json"))
# %%
args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': 'model/chinese_wwm_ext/bert_config.json',
    # 'bert_config_file_name': 'save_pretrained/weibo_x20_pretrained/Bert_2910/config.json',
    'hidden_dim': 300,
    'max_seq_length': 150,
    'max_scan_num': 1000000,
    'train_file': './data/weibonew/train_20_epoch.json',
    'eval_file': './data/weibo/dev.json',
    'test_file': './data/weibo/test.json',
    'bert_vocab_file': 'model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/weibo/labels.txt',
    'output_eval': True,
    'loader_name': 'le_loader',
    "word_embedding_file": "./data/tencent/word_embedding.txt",
    "word_vocab_file": "./data/tencent/tencent_vocab.txt",
    "default_tag": "O",
    'batch_size': 16,
    'eval_batch_size': 64,
    'do_shuffle': True,
    'model_name': 'LEBert',
    'task_name': 'weibo_20_epoch_GPT2',
    'pretrained_file_name':'model/chinese_wwm_ext/pytorch_model.bin'
    # 'pretrained_file_name':'save_pretrained/weibo_x20_pretrained/Bert_2910/pytorch_model.bin'
}

from CC.trainer import NERTrainer
trainer = NERTrainer(**args)

for i in trainer():
    a = i


# %%
from CC.loaders import *
from tqdm import *
from CC.loaders.utils import *
import json
from CC.pre_trained import NERPreTrainer

args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    'max_seq_length': 512,
    'max_scan_num': 1000000,
    'train_file': './data/weibonew/train_20.json',
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
    'task_name': 'weibo_x20_pretrained',
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

exit()
# %%
