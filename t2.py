#%%
from ner.loaders import LLoader

args = {
    "batch_size":4,
    "eval_batch_size":16,
    "test_file":"./data/test.json",
    "eval_file":"./data/dev.json",
    "train_file":"./data/train.json",
    "tag_file":"./data/labels.txt",
    "word_embedding_file":"./data/tencent/word_embedding.txt",
    "word_vocab_file":"./data/tencent/tencent_vocab.txt",
    "bert_vocab_file":"./data/bert/basechinese/vocab.txt",
    "default_tag":"O"
}

LLoader(**args).myData[0][4]
# %%
