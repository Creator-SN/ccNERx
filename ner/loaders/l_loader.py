from ner.loaders.utils import *
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import tensor
from transformers import BertTokenizer
from tqdm import *
from typing import *
from ICCSupervised.ICCSupervised import IDataLoader
import json
import numpy as np
import random


class LLoader(IDataLoader):
    def __init__(self, **args):
        self.read_data_set(**args)
        self.verify_data()
        self.process_data(args["batch_size"], args["eval_batch_size"])

    def read_data_set(self, **args):
        assert "word_embedding_file" in args, "argument word_embedding_file: required embeding file path"
        assert "word_vocab_file" in args, "argument word_vocab_file: required word vocab file to build lexicon tree"
        assert "train_file" in args, "argument train_file: required train file path"
        assert "eval_file" in args, "argument eval_file: required eval file path"
        assert "test_file" in args, "argument test_file: required test file path"
        assert "tag_file" in args, "argument tag_file: required label file path"
        # assert "config_name" in args, "argument config_name: required bert config file path"
        assert "bert_vocab_file" in args, "argument bert_vocab_file: required bert_vocab file path"
        self.output_eval = True
        if "output_eval" in args:
            self.output_eval = args["output_eval"]
        self.max_scan: int = 1000000
        if "max_scan_num" in args:
            self.max_scan = int(args["max_scan_num"])
        self.add_seq_vocab: bool = False
        if "add_seq_vocab" in args:
            self.add_seq_vocab = args["add_seq_vocab"]
        self.max_seq_length: int = 256
        if "max_seq_length" in args:
            self.max_seq_length: int = args["max_seq_length"]
        self.max_word_num = 5
        if "max_word_num" in args:
            self.max_word_num = args["max_word_num"]
        self.default_tag = "O"
        if "default_tag" in args:
            self.default_tag = args["default_tag"]

        # build lexicon tree
        self.lexicon_tree: Trie = TrieFactory.get_trie_from_vocabs(
            [args["word_vocab_file"]], self.max_scan)

        self.data_files: List[str] = [args["train_file"],
                                      args["eval_file"], args["test_file"]]

        self.matched_words: List[str] = TrieFactory.get_all_matched_word_from_dataset(
            self.data_files, self.lexicon_tree)

        self.word_vocab: Vocab = Vocab().from_list(
            self.matched_words, is_word=True, has_default=False, unk_num=5)

        self.tag_vocab: Vocab = Vocab().from_files(
            [args["tag_file"]], is_word=False)

        self.vocab_embedding, self.embedding_dim = VocabEmbedding(self.word_vocab).build_from_file(
            args["word_embedding_file"], self.max_scan, self.add_seq_vocab).get_embedding()

        self.tokenizer = BertTokenizer.from_pretrained(args["bert_vocab_file"])

    def verify_data(self):
        pass

    def process_data(self, batch_size: int, eval_batch_size: int = None):
        assert batch_size is not None, "argument batch_size: required"
        self.myData = LBertDataSet(self.data_files[0], self.tokenizer, self.lexicon_tree, self.word_vocab,
                                   self.tag_vocab, self.max_word_num, self.max_seq_length, self.default_tag)
        self.dataiter = DataLoader(self.myData, batch_size=batch_size)
        if self.output_eval:
            assert eval_batch_size is not None, "argument eval_batch_size: required"
            self.myData_eval = LBertDataSet(self.data_files[1], self.tokenizer, self.lexicon_tree, self.word_vocab,
                                            self.tag_vocab, self.max_word_num,  self.max_seq_length, self.default_tag)
            self.dataiter_eval = DataLoader(
                self.myData_eval, batch_size=eval_batch_size)

    def __call__(self):
        if self.output_eval:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'eval_set': self.myData_eval,
                'eval_iter': self.dataiter_eval,
                'vocab_embedding': self.vocab_embedding,
                'embedding_dim': self.embedding_dim,
                'word_vocab': self.word_vocab,
                'tag_vocab': self.tag_vocab
            }
        else:
            return {
                'train_set': self.myData,
                'train_iter': self.dataiter,
                'vocab_embedding': self.vocab_embedding,
                'embedding_dim': self.embedding_dim,
                'word_vocab': self.word_vocab,
                'tag_vocab': self.tag_vocab
            }


class LBertDataSet(Dataset):
    def __init__(self, file: str, tokenizer, lexicon_tree: Trie, word_vocab: Vocab, tag_vocab: Vocab, max_word_num: int, max_seq_length: int, default_tag: str, do_shuffle: bool = False):
        self.file: str = file
        self.tokenizer = tokenizer
        self.lexicon_tree: Trie = lexicon_tree
        self.word_vocab: Vocab = word_vocab
        self.label_vocab: Vocab = tag_vocab
        self.max_word_num: int = max_word_num
        self.max_seq_length: int = max_seq_length
        self.default_tag: str = default_tag
        self.do_shuffle: bool = do_shuffle
        self.__init_dataset()

    def __init_dataset(self):
        line_total = FileUtil.count_lines(self.file)
        self.input_token_ids = []
        self.segment_ids = []
        self.attention_mask = []
        self.matched_word_ids = []
        self.matched_word_mask = []
        self.labels = []

        for line in tqdm(FileUtil.line_iter(self.file), desc=f"load dataset from {self.file}", total=line_total):
            line = line.strip()
            data: Dict[str, List[Any]] = json.loads(line)
            assert "text" in data and "label" in data, "dataset json type error"
            text = ["[CLS]"] + data["text"][:self.max_seq_length-2] + ["[SEP]"]
            label = [self.default_tag] + \
                data["label"][:self.max_seq_length-2] + [self.default_tag]
            token_ids = self.tokenizer.convert_tokens_to_ids(text)
            label_ids = self.label_vocab.token2id(label)

            labels = np.zeros(self.max_seq_length, dtype=np.int)
            labels[:len(label_ids)] = label_ids
            # init input
            input_token_ids = np.zeros(self.max_seq_length, dtype=np.int)
            input_token_ids[:len(token_ids)] = token_ids
            segment_ids = np.ones(self.max_seq_length, dtype=np.int)
            segment_ids[:len(token_ids)] = 0
            attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
            attention_mask[:len(token_ids)] = 1
            matched_word_ids = np.zeros(
                (self.max_seq_length, self.max_word_num), dtype=np.int)
            matched_word_mask = np.zeros(
                (self.max_seq_length, self.max_word_num), dtype=np.int)
            # get matched word
            matched_words = self.lexicon_tree.getAllMatchedWordList(
                text, self.max_word_num)
            for i, words in enumerate(matched_words):
                word_ids = self.word_vocab.token2id(words)
                matched_word_ids[i][:len(word_ids)] = word_ids
                matched_word_mask[i][:len(word_ids)] = 1

            assert input_token_ids.shape[0] == segment_ids.shape[0]
            assert input_token_ids.shape[0] == attention_mask.shape[0]
            assert input_token_ids.shape[0] == matched_word_ids.shape[0]
            assert input_token_ids.shape[0] == matched_word_mask.shape[0]
            assert input_token_ids.shape[0] == labels.shape[0]
            assert matched_word_ids.shape[1] == matched_word_mask.shape[1]

            self.input_token_ids.append(input_token_ids)
            self.segment_ids.append(segment_ids)
            self.attention_mask.append(attention_mask)
            self.matched_word_ids.append(matched_word_ids)
            self.matched_word_mask.append(matched_word_mask)
            self.labels.append(labels)
        self.size = len(self.input_token_ids)
        self.input_token_ids = np.array(self.input_token_ids)
        self.segment_ids = np.array(self.segment_ids)
        self.attention_mask = np.array(self.attention_mask)
        self.matched_word_ids = np.array(self.matched_word_ids)
        self.matched_word_mask = np.array(self.matched_word_mask)
        self.labels = np.array(self.labels)
        self.indexes = [i for i in range(self.size)]
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        return {
            'input_ids': tensor(self.input_token_ids[idx]),
            'attention_mask': tensor(self.attention_mask[idx]),
            'token_type_ids': tensor(self.segment_ids[idx]),
            'matched_word_ids': tensor(self.matched_word_ids[idx]),
            'matched_word_mask': tensor(self.matched_word_mask[idx]),
            'labels': tensor(self.labels[idx])
        }

    def __len__(self):
        return self.size
