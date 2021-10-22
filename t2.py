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
    'train_file': './data/chinese_data/pre_train.json',
    'eval_file': './data/chinese_data/pre_dev.json',
    'test_file': './data/chinese_data/pre_test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    "bert_model_file": "./save_model/Super_LEBert/LEBert/LEBert_14910.pth",
    'tag_file': './data/chinese_data/tags_list.txt',
    'loader_name': 'le_loader',
    "word_embedding_file": "./data/tencent/word_embedding.txt",
    "word_vocab_file": "./data/tencent/tencent_vocab.txt",
    "default_tag": "O",
    'batch_size': 8,
    'eval_batch_size': 64,
    'model_name': 'LEBert',
    'task_name': 'Super_LEBert',
    'lstm_crf_model_file': './save_model/Super_LEBert/lstm_crf/lstm_crf_14910.pth',
    "use_gpu": True,
}

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
