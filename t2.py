# %%
from CC.loaders import *

args = {
    'num_epochs': 30,
    'num_gpus': [0, 1, 2, 3],
    'bert_config_file_name': './model/chinese_wwm_ext/config.json',
    'bert_pretrain_path': './model/chinese_wwm_ext/',
    'hidden_dim': 300,
    'max_seq_length': 150,
    'max_scan_num': 1000000,
    'train_file': './data/weibo/train.json',
    'eval_file': './data/weibo/dev.json',
    'test_file': './data/weibo/test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/weibo/pretrained_labels.txt',
    'external_entities_file':"data/entities_data_label.json",
    'loader_name': 'labellex_loader',
    "word_embedding_file": "./data/tencent/word_embedding.txt",
    "word_vocab_file": "./data/tencent/tencent_vocab.txt",
    "tag_embedding_file":"data/tencent/lebert_full_label_embedding_tx.txt",
    "default_tag": "O",
    'batch_size': 8,
    'eval_batch_size': 64,
    'model_name': 'LEXBert',
    'task_name': 'Loader_test',
    "use_gpu": True,
    "debug": True,
    "pass_none_rule": True,
    "tag_rules": {
        "O":"非实体",
        "PER.NOM": "指代人名",
        "LOC.NAM": "地名",
        "PER.NAM": "人名",
        "GPE.NAM": "政体",
        "ORG.NAM": "机构",
        "ORG.NOM": "指代机构",
        "LOC.NOM": "指代地名",
        "GPE.NOM": "指代政体",
        "NR":"人名",
        "NS":"地名",
        "NT":"组织机构",
        "CONT": "国家",
        "PRO":"职位",
        "RACE":"种族",
        "TITLE":"工作名称",
        "EDU":"教育经历",
        "NAME":"名字",
        "ORG": "机构",
        "LOC": "地名",
        "PER": "人名",
        "GPE": "政治实体",
        "Time": "时间",
        "Thing": "物品",
        "Metric": "度量",
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
    },
    "ignore_rules":[
        "name"
    ],
    "skip_single_matched_word":True,
}


loader = FTLoaderV1(**args)


#%%
old_loader = LXLoader(**args)

from CC.loaders.lex_loader_new import *
new_loader = LXLoader(**args)

#%%
print(len(old_loader.myData))
choices = ("input_ids","origin_labels","input_labels","labels")
for i in range(len(old_loader.myData)):
    flag = False
    for j in range(len(choices)):
        a = old_loader.myData[i][choices[j]]
        b = new_loader.myData[i][choices[j]]
        if not torch.equal(a,b):
            flag = True
            break
    if flag:
        for j in range(len(choices)):
            a = old_loader.myData[i][choices[j]].tolist()
            b = new_loader.myData[i][choices[j]].tolist()
            print(choices[j])
            if j<3:
                print("old",old_loader.tokenizer.decode(a))
                print("new",new_loader.tokenizer.decode(b))
            else:
                print("old",old_loader.tag_vocab.id2token(a))
                print("new",new_loader.tag_vocab.id2token(b))
            print("----------------")
        input()
        from IPython.display import clear_output
        clear_output()
    


# %%
loader.myData[1]["matched_label_embedding"]
#%%
choices = ("input_ids","origin_labels","input_labels","labels","matched_label_ids","matched_word_ids")
index = 4
for i in loader.myData[0:1][choices[index]].tolist():
    if index<3:
        print(loader.tokenizer.decode(i))
    elif index==3:
        print(loader.tag_vocab.id2token(i))
    elif index==4:
        print(loader.entity_tag_vocab.id2token(i))
    else:
        print(loader.word_vocab.id2token(i))

#%%
import os
files = os.listdir("/tmp/tmp0pu1kou7")
files = sorted(files)
import pickle
for file in files:
    with open(os.path.join("/tmp/tmp0pu1kou7",file),"rb") as f:
        v = pickle.load(f)
        value = v.max()
        if (value>0):
            for i in v:
                if i.max()>0:
                    print(i)
            raise



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
from tools.expand_data import DataExpand
loader = DataExpand(**{
    "debug": True,
    "file_name": "data/weibo/train.json",
    "allow_origin": False
}).read_data_set("data/weibo/train.json", 1.0) \
    .process_data(20) \
    .to_file("./data/weibonew/train_test.json")

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
    
# %%
from CC.loaders.utils import LabelCounter
counter = LabelCounter()
example = {"text": ["延", "参", "法", "师", "品", "味", "人", "生", "如", "同", "走", "进", "一", "片", "山", "水", "，", "静", "静", "的", "呼", "吸", "，", "安", "静", "的", "欣", "赏", "，", "这", "就", "是", "生", "活", "。"], "label": ["B-PER.NAM", "E-PER.NAM", "B-PER.NOM", "E-PER.NOM", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
counter.add(example["label"],example["text"])
counter.pick("PER.NOM","人民")
# %%
print(counter.label_counter.keys())

# %%
from tools.tag_embedding_gen import *
rules = {
    "O":"非实体",
    "PER.NOM": "指代人名",
    "LOC.NAM": "地名",
    "PER.NAM": "人名",
    "GPE.NAM": "政体",
    "ORG.NAM": "机构",
    "ORG.NOM": "指代机构",
    "LOC.NOM": "指代地名",
    "GPE.NOM": "指代政体",
    "ORG": "机构",
    "LOC": "地名",
    "PER": "人名",
    "Time": "时间",
    "Thing": "物品",
    "Metric": "度量",
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
gen = EmbeddingGenerator()
for key in rules:
    gen+=(key,rules[key])
gen.to_file("data/tencent/label_embedding.txt")
        
# %%
