# %%
from CC.loaders import *
import pickle
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
    'max_scan_num': 100,
    'train_file': './data/SuperNER/pre_train.json',
    'eval_file': './data/SuperNER/pre_dev.json',
    'test_file': './data/SuperNER/pre_test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    "bert_model_file": "./save_model/weibo/LEBert/LEBert_845.pth",
    'tag_file': './data/SuperNER/tags_list.txt',
    'loader_name': 'le_loader',
    "word_embedding_file": "./data/tencent/word_embedding.txt",
    "word_vocab_file": "./data/tencent/tencent_vocab.txt",
    "word_vocab_file_with_tag": "./data/tencent/tencent_vocab_with_tag.json",
    "default_tag": "O",
    'batch_size': 8,
    'eval_batch_size': 64,
    'model_name': 'LEXBert',
    'task_name': 'Super_LEXBert',
    'lstm_crf_model_file': './save_model/Super_LEBert/lstm_crf/lstm_crf_14910.pth',
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

# %%
loader = LXLoader(**args)


#%% 
loader.myData[0]["input_ids"]
loader.tokenizer.decode(loader.myData[0]["input_ids"])

#%%
loader.myData[0]["labels"]
loader.tag_vocab.id2token(loader.myData[0]["labels"].tolist())

#%%
loader.tokenizer.decode(loader.myData[0]["input_labels"])

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

lexicon_tree: Trie = TrieFactory.get_trie_from_vocabs(
    ["data/tencent/tencent_vocab.txt"])
# %%


def x(): return TrieFactory.get_trie_from_vocabs(
    ["data/tencent/tencent_vocab.txt"])


x()
# %%
with open("./temp/lexicon_tree_cache.pkl", "wb") as f:
    pickle.dump(lexicon_tree, f)
# %%
with open("./temp/lexicon_tree_cache.pkl", "rb") as f:
    lexicon_tree = pickle.load(f)

lexicon_tree.search("我们")
# %%


class A():
    def __init__(self):
        self.a = []

    def __add__(self, a):
        if isinstance(a, tuple):
            for i in a:
                self.a += [i]
        else:
            self.a += [a]
        return self


class B(A):
    def __init__(self):
        super().__init__()


a = B()
a += 1
a += 2
a += (1, 2)
print(a.__dict__)
# %%


def add(a):
    print(type(a))
    return a


a = {
    "a": 2
}

print(add(**a))


# %%

a = [1, 2, 3]
b = [2, 3, 4]

for a, b in zip(a, b):
    print(a, b)
# %%
from transformers import BertTokenizer
bt = BertTokenizer.from_pretrained("./data/bert/chinese_wwm_ext/")
# %%
bt("a sentences. Hello world!")
# %%
